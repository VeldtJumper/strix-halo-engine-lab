#include "strix/cost_table_cache.hpp"
#include "strix/model_content_hash.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cerrno>
#include <cstdlib>
#include <random>
#include <system_error>
#include <utility>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#include <process.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
extern char** environ;
#endif

namespace strix {
namespace {
constexpr std::string_view magic="strix-cost-table";

std::FILE* open_file(const std::filesystem::path& path,bool write) {
#ifdef _WIN32
    std::FILE* f=nullptr;
    return _wfopen_s(&f,path.c_str(),write?L"wb":L"rb")==0?f:nullptr;
#else
    return std::fopen(path.c_str(),write?"wb":"rb");
#endif
}
unsigned long process_id() noexcept {
#ifdef _WIN32
    return static_cast<unsigned long>(_getpid());
#else
    return static_cast<unsigned long>(::getpid());
#endif
}
bool sync_file(std::FILE* f) noexcept {
#ifdef _WIN32
    return _commit(_fileno(f))==0;
#else
    return ::fsync(::fileno(f))==0;
#endif
}
// Atomic replacement of `to` by `from` (same directory), the old file intact on failure.
bool replace_file(const std::filesystem::path& from,const std::filesystem::path& to) noexcept {
#ifdef _WIN32
    return MoveFileExW(from.c_str(),to.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)!=0;
#else
    return ::rename(from.c_str(),to.c_str())==0;
#endif
}
// Exclusive advisory lock on a per-key lock file (blocking; released by the destructor).
class FileLock {
public:
    explicit FileLock(const std::filesystem::path& path) {
#ifdef _WIN32
        handle_=CreateFileW(path.c_str(),GENERIC_READ|GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,nullptr,
            OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
        if(handle_==INVALID_HANDLE_VALUE) return;
        OVERLAPPED overlapped{};
        held_=LockFileEx(handle_,LOCKFILE_EXCLUSIVE_LOCK,0,MAXDWORD,MAXDWORD,&overlapped)!=0;
#else
        fd_=::open(path.c_str(),O_RDWR|O_CREAT|O_CLOEXEC,0644);
        if(fd_<0) return;
        int r=0;
        do r=::flock(fd_,LOCK_EX); while(r!=0 && errno==EINTR);
        held_=r==0;
#endif
    }
    ~FileLock() {
#ifdef _WIN32
        if(handle_!=INVALID_HANDLE_VALUE) {
            if(held_) {OVERLAPPED overlapped{};UnlockFileEx(handle_,0,MAXDWORD,MAXDWORD,&overlapped);}
            CloseHandle(handle_);
        }
#else
        if(fd_>=0) {if(held_) ::flock(fd_,LOCK_UN);::close(fd_);}
#endif
    }
    FileLock(const FileLock&)=delete;
    FileLock& operator=(const FileLock&)=delete;
    [[nodiscard]] bool held() const noexcept {return held_;}
private:
#ifdef _WIN32
    HANDLE handle_=INVALID_HANDLE_VALUE;
#else
    int fd_=-1;
#endif
    bool held_=false;
};

std::uint64_t fnv1a64(std::string_view bytes) noexcept {
    std::uint64_t h=1469598103934665603ULL;
    for(const char c:bytes) {h^=static_cast<unsigned char>(c);h*=1099511628211ULL;}
    return h;
}
std::string hex64(std::uint64_t v) {
    static constexpr char digits[]="0123456789abcdef";
    std::string out(16,'0');
    for(std::size_t i=16;i-->0;v>>=4U) out[i]=digits[v&15U];
    return out;
}
// Key values are single tokens: bytes outside 0x21..0x7e and '%' are percent-encoded.
std::string encode(std::string_view value) {
    static constexpr char digits[]="0123456789ABCDEF";
    std::string out;
    for(const char c:value) {
        const auto u=static_cast<unsigned char>(c);
        if(u<0x21U || u>0x7eU || c=='%') {out+='%';out+=digits[u>>4U];out+=digits[u&15U];}
        else out+=c;
    }
    return out;
}
std::string number(double v) {
    std::array<char,64> buffer{};
    const auto r=std::to_chars(buffer.data(),buffer.data()+buffer.size(),v);
    return std::string(buffer.data(),r.ptr);
}
template<class T> bool parse(std::string_view text,T& out) {
    if(text.empty()) return false;
    const auto r=std::from_chars(text.data(),text.data()+text.size(),out);
    return r.ec==std::errc{} && r.ptr==text.data()+text.size();
}
bool usable(const CostTableEntry& e) noexcept {return std::isfinite(e.ms) && e.ms>0.0 && e.samples>0U;}
std::string canonical_key(const CostTableKey& k) {
    return encode(k.engine)+'\n'+encode(k.binary)+'\n'+encode(k.device)+'\n'+encode(k.model)+'\n'+encode(k.policy)+'\n'+
        std::to_string(k.context_bucket);
}
bool valid_engine(std::string_view engine) noexcept {
    if(engine.empty() || engine.size()>64U) return false;
    return std::all_of(engine.begin(),engine.end(),[](char c){
        return (c>='a' && c<='z') || (c>='A' && c<='Z') || (c>='0' && c<='9') || c=='_' || c=='.' || c=='-';});
}
// Splits `body` into lines; false when it does not end with a newline.
bool split_lines(std::string_view body,std::vector<std::string_view>& lines) {
    if(body.empty() || body.back()!='\n') return false;
    std::size_t at=0;
    while(at<body.size()) {const auto end=body.find('\n',at);lines.push_back(body.substr(at,end-at));at=end+1U;}
    return true;
}
bool field(std::string_view line,std::string_view name,std::string_view& value) {
    if(line.size()<=name.size()+1U || line.substr(0,name.size())!=name || line[name.size()]!=' ') return false;
    value=line.substr(name.size()+1U);
    return value.find(' ')==std::string_view::npos;
}
} // namespace

const char* cost_table_load_name(CostTableLoad s) noexcept {
    switch(s) {
    case CostTableLoad::not_loaded: return "not-loaded";
    case CostTableLoad::missing: return "missing";
    case CostTableLoad::loaded: return "loaded";
    case CostTableLoad::corrupt: return "corrupt";
    case CostTableLoad::version: return "version";
    case CostTableLoad::mismatch: return "mismatch";
    }
    return "unknown";
}
std::uint32_t cost_table_context_bucket(std::uint64_t position) noexcept {
    std::uint32_t bucket=0;
    for(std::uint64_t edge=8192U;position>=edge && bucket<40U;edge*=2U) ++bucket;
    return bucket;
}
bool cost_table_directory_admitted(const std::filesystem::path& directory) {
    if(directory.empty()) return false;
    std::error_code ec;
    auto path=std::filesystem::absolute(directory,ec);
    if(ec) return false;
    path=path.lexically_normal();
    // Never inside a source checkout: a table is per machine, never source.
    for(auto at=path;;at=at.parent_path()) {
        if(std::filesystem::exists(at/".git",ec)) return false;
        if(at==at.parent_path() || at.empty()) break;
    }
    return true;
}
std::optional<std::filesystem::path> cost_table_directory_from_env() {
    const char* v=std::getenv("STRIX_COST_TABLE_DIR");
    if(!v || !*v) return std::nullopt;
    std::filesystem::path path(v);
    if(!cost_table_directory_admitted(path)) {
        static std::atomic<bool> warned{false};
        if(!warned.exchange(true)) std::fprintf(stderr,"cost-table: STRIX_COST_TABLE_DIR refused (empty or inside a source checkout); off\n");
        return std::nullopt;
    }
    return path;
}
CostTableOptions cost_table_options_from_env() {
    CostTableOptions out;
    double v=0.0;
    if(const char* s=std::getenv("STRIX_COST_TABLE_RECHECK"); s && parse(std::string_view(s),v) && v>=0.0 && v<=1.0)
        out.recheck_fraction=v;
    if(const char* s=std::getenv("STRIX_COST_TABLE_DRIFT"); s && parse(std::string_view(s),v) && std::isfinite(v) && v>0.0)
        out.drift_tolerance=v;
    return out;
}
std::string cost_table_hash_hex(std::string_view bytes) {
    return model_content_identity_hex(sha256_bytes(std::as_bytes(std::span(bytes.data(),bytes.size()))));
}
std::string cost_table_executable_identity() {
    static const std::string identity=[]()->std::string {
        try {
#ifdef _WIN32
            std::wstring path(512,L'\0');
            for(;;) {
                const auto n=GetModuleFileNameW(nullptr,path.data(),static_cast<DWORD>(path.size()));
                if(n==0) return {};
                if(n<path.size()) {path.resize(n);break;}
                if(path.size()>=32768U) return {};
                path.resize(path.size()*2U);
            }
            return model_content_identity_hex(sha256_file_content(std::filesystem::path(path)));
#else
            return model_content_identity_hex(sha256_file_content("/proc/self/exe"));
#endif
        } catch(...) {return {};}
    }();
    return identity;
}
std::string cost_table_policy_identity(std::string_view configuration) {
    std::vector<std::string> env;
#ifdef _WIN32
    auto p=_environ;
#else
    auto p=environ;
#endif
    for(;p && *p;++p) {
        const std::string_view value(*p);
        if(value.starts_with("STRIX_COST_TABLE_") || value.starts_with("STRIX_CYCLE_LOG=") || value.starts_with("GPU_TURN_") ||
            value.starts_with("GPU_WINDOW=")) continue;
        for(const auto prefix:{"STRIX_","HIP_","HSA_","AMD_","ROCR_","GPU_"})
            if(value.starts_with(prefix)) {env.emplace_back(value);break;}
    }
    std::sort(env.begin(),env.end());
    std::string bytes="strix-cost-policy-v1\n";
    bytes+=std::to_string(configuration.size())+':';bytes.append(configuration);
    for(const auto& v:env) {bytes+=std::to_string(v.size())+':';bytes+=v;}
    return cost_table_hash_hex(bytes);
}

bool CostTableCache::valid_name(std::string_view name) noexcept {
    if(name.empty() || name.size()>64U) return false;
    return std::all_of(name.begin(),name.end(),[](char c){
        return (c>='a' && c<='z') || (c>='A' && c<='Z') || (c>='0' && c<='9') || c=='_' || c=='.' || c=='-' || c==':';});
}
CostTableCache::CostTableCache(std::filesystem::path directory,CostTableKey key,CostTableOptions options)
    :directory_(std::move(directory)),key_(std::move(key)),options_(options) {
    if(!(options_.recheck_fraction>=0.0 && options_.recheck_fraction<=1.0)) options_.recheck_fraction=0.15;
    if(!(std::isfinite(options_.drift_tolerance) && options_.drift_tolerance>0.0)) options_.drift_tolerance=0.15;
    if(valid_engine(key_.engine))
        file_=directory_/(key_.engine+'-'+cost_table_hash_hex(canonical_key(key_)).substr(0,32)+".costs");
}
std::string CostTableCache::serialize(const CostTableKey& k,std::uint64_t generation,std::uint64_t epoch,const Entries& entries) {
    std::string body=std::string(magic)+' '+std::to_string(format_version)+'\n';
    body+="engine "+encode(k.engine)+'\n';
    body+="binary "+encode(k.binary)+'\n';
    body+="device "+encode(k.device)+'\n';
    body+="model "+encode(k.model)+'\n';
    body+="policy "+encode(k.policy)+'\n';
    body+="bucket "+std::to_string(k.context_bucket)+'\n';
    body+="generation "+std::to_string(generation)+'\n';
    body+="epoch "+std::to_string(epoch)+'\n';
    std::size_t n=0;
    for(const auto& [name,e]:entries) if(valid_name(name) && usable(e)) ++n;
    body+="entries "+std::to_string(n)+'\n';
    for(const auto& [name,e]:entries)
        if(valid_name(name) && usable(e)) body+="entry "+name+' '+number(e.ms)+' '+std::to_string(e.samples)+'\n';
    body+="checksum "+hex64(fnv1a64(body))+'\n';
    return body;
}
CostTableCache::Parsed CostTableCache::parse_body(std::string_view body,const CostTableKey& key) {
    Parsed out;
    const auto fail=[&](CostTableLoad s){out.entries.clear();out.generation=out.epoch=0;out.status=s;return out;};
    std::vector<std::string_view> lines;
    if(!split_lines(body,lines) || lines.size()<12U) return fail(CostTableLoad::corrupt);
    // Version first: an other-version file need not follow this layout.
    std::string_view v;std::uint32_t version=0;
    if(!field(lines[0],magic,v) || !parse(v,version)) return fail(CostTableLoad::corrupt);
    if(version!=format_version) return fail(CostTableLoad::version);
    const auto& last=lines.back();std::string_view sum;
    if(!field(last,"checksum",sum) || sum.size()!=16U) return fail(CostTableLoad::corrupt);
    const auto covered=body.substr(0,body.size()-last.size()-1U);
    if(hex64(fnv1a64(covered))!=sum) return fail(CostTableLoad::corrupt);
    const std::array<std::pair<std::string_view,std::string>,5> keys{{
        {"engine",encode(key.engine)},{"binary",encode(key.binary)},{"device",encode(key.device)},
        {"model",encode(key.model)},{"policy",encode(key.policy)}}};
    bool mismatch=false;
    for(std::size_t k=0;k<keys.size();++k) {
        if(!field(lines[1U+k],keys[k].first,v)) return fail(CostTableLoad::corrupt);
        mismatch=mismatch || v!=keys[k].second;
    }
    std::uint32_t bucket=0;std::uint64_t n=0;
    if(!field(lines[6],"bucket",v) || !parse(v,bucket)) return fail(CostTableLoad::corrupt);
    if(!field(lines[7],"generation",v) || !parse(v,out.generation)) return fail(CostTableLoad::corrupt);
    if(!field(lines[8],"epoch",v) || !parse(v,out.epoch)) return fail(CostTableLoad::corrupt);
    if(!field(lines[9],"entries",v) || !parse(v,n) || n!=lines.size()-11U) return fail(CostTableLoad::corrupt);
    for(std::size_t i=10;i+1U<lines.size();++i) {
        const auto line=lines[i];
        if(!line.starts_with("entry ")) return fail(CostTableLoad::corrupt);
        const auto rest=line.substr(6);
        const auto a=rest.find(' ');
        if(a==std::string_view::npos) return fail(CostTableLoad::corrupt);
        const auto b=rest.find(' ',a+1U);
        if(b==std::string_view::npos || rest.find(' ',b+1U)!=std::string_view::npos) return fail(CostTableLoad::corrupt);
        const auto name=rest.substr(0,a);
        CostTableEntry e;
        if(!valid_name(name) || !parse(rest.substr(a+1U,b-a-1U),e.ms) || !parse(rest.substr(b+1U),e.samples) || !usable(e) ||
            !out.entries.emplace(std::string(name),e).second) return fail(CostTableLoad::corrupt);
    }
    // Never trust a table of another key, even a well-formed one.
    if(mismatch || bucket!=key.context_bucket) return fail(CostTableLoad::mismatch);
    out.status=CostTableLoad::loaded;
    return out;
}
CostTableCache::Parsed CostTableCache::read_file(const std::filesystem::path& path,const CostTableKey& key) {
    try {
        std::FILE* f=open_file(path,false);
        if(!f) {Parsed out;out.status=CostTableLoad::missing;return out;}
        // At most limit + 1 bytes: an oversized file is corrupt without being read whole.
        std::string body(max_file_bytes+1U,'\0');
        const auto got=std::fread(body.data(),1U,body.size(),f);
        const bool error=std::ferror(f)!=0;
        std::fclose(f);
        if(error || got>max_file_bytes) {Parsed out;out.status=CostTableLoad::corrupt;return out;}
        body.resize(got);
        return parse_body(body,key);
    } catch(...) {
        Parsed out;out.status=CostTableLoad::corrupt;return out;
    }
}
CostTableLoad CostTableCache::load() {
    if(status_!=CostTableLoad::not_loaded) return status_;
    if(file_.empty() || key_.binary.empty()) return status_=CostTableLoad::mismatch;
    auto parsed=read_file(file_,key_);
    status_=parsed.status;
    if(status_!=CostTableLoad::loaded) return status_;
    loaded_=std::move(parsed.entries);generation_=parsed.generation;epoch_=parsed.epoch;
    // Rotating drift check: ceil(fraction * n) entries (at least one), a different window each generation.
    const auto count=loaded_.size();
    if(count && options_.recheck_fraction>0.0) {
        const auto k=std::min<std::size_t>(count,std::max<std::size_t>(1U,
            static_cast<std::size_t>(std::ceil(options_.recheck_fraction*static_cast<double>(count)-1e-9))));
        std::vector<std::string> names;
        for(const auto& [name,e]:loaded_) names.push_back(name);
        const auto start=static_cast<std::size_t>((generation_%count)*k%count);
        for(std::size_t i=0;i<k;++i) recheck_.insert(names[(start+i)%count]);
    }
    return status_;
}
std::optional<CostTableEntry> CostTableCache::warm(std::string_view name) const {
    if(const auto m=measured_.find(name);m!=measured_.end()) return m->second;
    if(drifted_ || recheck_.count(name)) return std::nullopt;
    if(const auto l=loaded_.find(name);l!=loaded_.end()) return l->second;
    return std::nullopt;
}
bool CostTableCache::rechecking(std::string_view name) const {
    return !drifted_ && recheck_.count(name) && !measured_.count(name);
}
void CostTableCache::record(std::string_view name,CostTableEntry measured) {
    if(!valid_name(name) || !usable(measured)) return;
    if(rechecking(name)) {
        const auto cached=loaded_.find(name)->second.ms;
        if(std::abs(measured.ms/cached-1.0)>options_.drift_tolerance) {
            drifted_=true;
            std::fprintf(stderr,"cost-table: drift on %s (%s): cached %.3f ms, measured %.3f ms; table withdrawn\n",
                key_.engine.c_str(),std::string(name).c_str(),cached,measured.ms);
        }
    }
    measured_.insert_or_assign(std::string(name),measured);
    dirty_=true;
}
bool CostTableCache::store() {
    if(file_.empty() || key_.binary.empty() || !dirty_) return false;
    try {
        std::error_code ec;
        std::filesystem::create_directories(directory_,ec);
        FileLock lock(std::filesystem::path(file_.string()+".lock"));
        if(!lock.held()) return false;
        // Merge into the file as it is now, never into this process's (possibly older) snapshot.
        const auto current=read_file(file_,key_);
        const bool valid=current.status==CostTableLoad::loaded;
        const auto disk_epoch=valid?current.epoch:0U;
        // Another writer withdrew the table we loaded: stop using our copy of it too.
        if(valid && current.epoch>epoch_) {loaded_.clear();recheck_.clear();}
        Entries table;
        std::uint64_t epoch=std::max(disk_epoch,epoch_);
        if(drifted_ && (!valid || current.epoch<=epoch_)) {
            // Our drift check judged the table we loaded (or a newer one in the same epoch): withdraw it.
            epoch+=1U;
        } else if(valid) {
            // Not drifted, or another writer already started a newer epoch: keep the file's entries.
            table=current.entries;
        }
        for(const auto& [name,e]:measured_) table.insert_or_assign(name,e);
        if(table.empty()) return false;
        const auto generation=std::max(valid?current.generation:0U,generation_)+1U;
        const auto body=serialize(key_,generation,epoch,table);
        static std::atomic<std::uint64_t> counter{0};
        std::random_device rd;
        const auto temp=std::filesystem::path(file_.string()+".tmp."+std::to_string(process_id())+'.'+
            std::to_string(counter.fetch_add(1U))+'.'+hex64(rd()));
        bool ok=false;
        if(std::FILE* f=open_file(temp,true)) {
            ok=std::fwrite(body.data(),1U,body.size(),f)==body.size() && std::fflush(f)==0 && sync_file(f);
            ok=std::fclose(f)==0 && ok;
        }
        ok=ok && replace_file(temp,file_);
        if(!ok) {std::filesystem::remove(temp,ec);return false;}
        generation_=generation;epoch_=epoch;dirty_=false;
        // The withdrawal is now on disk: consume it (a later store must not bump the epoch again and delete newer
        // entries), and keep the withdrawn priors unusable in this process.
        if(drifted_) {drifted_=false;loaded_.clear();recheck_.clear();}
        return true;
    } catch(...) {
        return false;
    }
}
} // namespace strix
