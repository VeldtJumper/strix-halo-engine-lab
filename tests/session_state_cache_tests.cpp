#include "strix/session_state_cache.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/unbuffered_file.hpp"
#include "state_file_fixtures.inc"
#include "../apps/hip_decode_options.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <cstdlib>
#include <iterator>
#include <optional>
#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/vfs.h>
#include <unistd.h>
#endif

namespace {
void check(bool b,const char* why) { if(!b) throw std::runtime_error(why); }
template<class F> void refused(F f,const char* why) {
    bool threw=false; try { f(); } catch(const std::exception&) { threw=true; } check(threw,why);
}
strix::PrefixCacheBuffer heap(std::uint64_t n) {
    return {std::shared_ptr<std::byte>(new std::byte[static_cast<std::size_t>(n?n:1)],std::default_delete<std::byte[]>()),n};
}
strix::ModelContentIdentity hash(std::string_view s) { return strix::sha256_bytes(std::as_bytes(std::span(s.data(),s.size()))); }
strix::PrefixCacheBuffer clone(const strix::PrefixCacheBuffer& b) {
    auto c=heap(b.bytes); std::memcpy(c.data.get(),b.data.get(),static_cast<std::size_t>(b.bytes)); return c;
}
void reroot(strix::PrefixCacheBuffer& b) {
    auto h=strix::sha256_bytes_fast(b.span().first(static_cast<std::size_t>(b.bytes-32)));
    std::memcpy(b.data.get()+b.bytes-32,h.bytes.data(),32);
}
}
void session_tail_plan_test() {
    const char* previous=std::getenv("STRIX_QPF2K_SCHED");
    const std::optional<std::string> saved=previous?std::optional<std::string>{previous}:std::nullopt;
    const auto set=[](const char* v) {
#ifdef _WIN32
        check(_putenv_s("STRIX_QPF2K_SCHED",v?v:"")==0,"tail environment update");
#else
        check((v?setenv("STRIX_QPF2K_SCHED",v,1):unsetenv("STRIX_QPF2K_SCHED"))==0,"tail environment update");
#endif
    };
    set("8");
    const auto make=[](std::size_t n) {
        strix::decode_cli::Options o; o.known_prefix_window=true; o.known_prefix_prefill_explicit=true;
        o.known_prefix_full_rows=true; o.known_prefix_prefill_rows=2048; o.tokens.resize(n); return o;
    };
    for(int mode=0;mode<6;++mode) {
        auto a=make(32768), b=make(77);
        for(auto* o:{&a,&b}) {
            if(mode==0) o->session_save="checkpoint";
            if(mode==1) o->session_load="checkpoint";
            if(mode==2) o->session_live_control=true;
            if(mode==3) o->session_then_prompt="suffix";
            if(mode==4) o->session_logits="logits";
            if(mode==5) o->session_options_requested=true;
            strix::decode_cli::apply_default_known_prefill_tails(*o);
        }
        check(a.known_prefix_tail_rows==b.known_prefix_tail_rows && a.known_prefix_tail_rows.size()==4,"session tail plan depends on input length");
    }
    auto a=make(32768), b=make(77);
    strix::decode_cli::apply_default_known_prefill_tails(a);
    strix::decode_cli::apply_default_known_prefill_tails(b);
    check(a.known_prefix_tail_rows!=b.known_prefix_tail_rows,"non-session pruning changed");
    set(saved?saved->c_str():nullptr);
}
namespace {
void set_env(const char* key,const char* value) {
#ifdef _WIN32
    check(_putenv_s(key,value?value:"")==0,"environment update");
#else
    check((value?setenv(key,value,1):unsetenv(key))==0,"environment update");
#endif
}
struct EnvGuard {
    const char* key; std::optional<std::string> saved;
    explicit EnvGuard(const char* k) : key(k) { if(const char* v=std::getenv(k)) saved=v; }
    ~EnvGuard() { set_env(key,saved?saved->c_str():nullptr); }
};
std::vector<char> file_bytes(const std::filesystem::path& p) {
    std::ifstream in(p,std::ios::binary); return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};
}
#if defined(__linux__)
// Resident pages of a file (mincore over a read-only mapping); {resident, total}.
std::pair<std::uint64_t,std::uint64_t> resident_pages(const std::filesystem::path& p) {
    const int fd=::open(p.c_str(),O_RDONLY|O_CLOEXEC); check(fd>=0,"residency open");
    const auto size=static_cast<std::size_t>(std::filesystem::file_size(p));
    const auto page=static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    void* map=::mmap(nullptr,size,PROT_READ,MAP_SHARED,fd,0); ::close(fd);
    check(map!=MAP_FAILED,"residency mmap");
    std::vector<unsigned char> v((size+page-1)/page);
    const int rc=::mincore(map,size,v.data()); ::munmap(map,size); check(rc==0,"mincore");
    std::uint64_t r=0; for(auto b:v) r+=b&1U;
    return {r,static_cast<std::uint64_t>(v.size())};
}
// On tmpfs the page cache is the file: residency there says nothing about write-through.
bool page_cache_backed(const std::filesystem::path& p) {
    struct statfs fs{}; return ::statfs(p.c_str(),&fs)==0 && fs.f_type!=0x01021994;
}
#endif
strix::SessionCheckpoint large_checkpoint(const strix::SessionCompatibility& compat,std::uint64_t bytes,unsigned seed) {
    strix::SessionCheckpoint c; c.compatibility=compat;
    c.frontier.consumed=2; c.frontier.accepted=2; c.frontier.emitted=1; c.frontier.tokens={5,6};
    auto payload=heap(bytes);
    std::uint64_t x=0x9e3779b97f4a7c15ULL*seed+1;
    for(std::uint64_t i=0;i<bytes;++i) { x^=x<<13; x^=x>>7; x^=x<<17; payload.data.get()[i]=static_cast<std::byte>(x&255U); }
    c.sections.push_back({"target.kv",payload}); c.sections.push_back({"target.fixed",heap(0)});
    return c;
}
}
// A session.bin written before unbuffered I/O restores in every read mode, and today's
// encoder and both writers still produce exactly those bytes.
void old_format_session_tests(const std::filesystem::path& root) {
    const auto h=[](const char* s) { return strix::sha256_bytes(std::as_bytes(std::span(s,std::strlen(s)))); };
    strix::SessionCompatibility compat{h("fixture model"),h("fixture exe"),"fixture numerical config v1",true};
    strix::SessionCheckpoint c; c.compatibility=compat;
    c.frontier.consumed=5; c.frontier.accepted=5; c.frontier.emitted=3; c.frontier.predictor=4; c.frontier.pending=77;
    c.frontier.tokens={11,22,33,44,55};
    auto kv=heap(5000); for(std::uint64_t i=0;i<kv.bytes;++i) kv.data.get()[i]=static_cast<std::byte>((i*29+7)&255);
    auto fixed=heap(333); for(std::uint64_t i=0;i<fixed.bytes;++i) fixed.data.get()[i]=static_cast<std::byte>((i*13+1)&255);
    c.sections.push_back({"target.kv",kv}); c.sections.push_back({"target.fixed",fixed}); c.sections.push_back({"mtp.empty",heap(0)});
    const std::vector<char> old(reinterpret_cast<const char*>(old_format_session_bin),
        reinterpret_cast<const char*>(old_format_session_bin)+sizeof(old_format_session_bin));
    const auto blob=strix::encode_session_checkpoint(c,heap);
    check(blob.bytes==old.size() && !std::memcmp(blob.data.get(),old.data(),old.size()),"session encoding drifted from the old format");
    std::filesystem::create_directories(root/"old");
    { std::ofstream out(root/"old"/"session.bin",std::ios::binary); out.write(old.data(),static_cast<std::streamsize>(old.size())); }
    EnvGuard d("STRIX_SESSION_DIRECT_IO"), a("STRIX_SESSION_READAHEAD"), k("STRIX_SESSION_CHUNKED");
    for(const char* direct:{"0","1"}) for(const char* chunked:{"0","1"}) {
        set_env("STRIX_SESSION_DIRECT_IO",direct); set_env("STRIX_SESSION_CHUNKED",chunked);
        strix::SessionCheckpoint back;
        (void)strix::read_session_checkpoint(root/"old",compat,heap,old.size(),nullptr,&back);
        check(back.frontier.tokens==c.frontier.tokens && back.frontier.pending==77U && back.frontier.predictor==4 &&
            back.sections.size()==3 && back.sections[0].payload.bytes==5000 &&
            !std::memcmp(back.sections[0].payload.data.get(),kv.data.get(),5000) &&
            !std::memcmp(back.sections[1].payload.data.get(),fixed.data.get(),333),"old-format session restore differs");
        strix::publish_session_checkpoint(root/"new",blob);
        check(file_bytes(root/"new"/"session.bin")==old,"publication no longer writes the old-format bytes");
    }
    std::filesystem::remove_all(root);
}
// Chunked reader with and without read-ahead: same bytes for any read sizes, truncation
// refused, growth after the opened size detected.
void sequential_reader_tests(const std::filesystem::path& root) {
#if defined(__linux__)
    namespace io=strix::state_file_io;
    std::filesystem::create_directories(root);
    const auto file=root/"stream.bin";
    const std::uint64_t size=3*io::chunk_bytes+12345;
    std::vector<char> data(static_cast<std::size_t>(size));
    std::uint64_t x=88172645463325252ULL;
    for(auto& c:data) { x^=x<<13; x^=x>>7; x^=x<<17; c=static_cast<char>(x&255U); }
    { std::ofstream out(file,std::ios::binary); out.write(data.data(),static_cast<std::streamsize>(data.size())); }
    for(const bool ahead:{false,true}) for(const std::uint64_t piece:{std::uint64_t{1}<<20,std::uint64_t{4096*3+17},size}) {
        io::SequentialReader in(file,ahead);
        check(in.open() && in.size()==size && in.reading_ahead()==ahead,"reader open");
        std::vector<char> back(data.size());
        for(std::uint64_t at=0;at<size;at+=piece) in.read(back.data()+at,std::min(piece,size-at));
        check(back==data && !in.grew(),"chunked read differs");
        refused([&]{char c; in.read(&c,1);},"read past the end accepted");
    }
    {
        io::SequentialReader in(file,true);
        std::vector<char> part(1000); in.read(part.data(),part.size());
    }  // destroyed while the helper reads ahead
    {
        io::SequentialReader in(file,false);
        { std::ofstream grow(file,std::ios::binary|std::ios::app); grow<<"more"; }
        std::vector<char> back(data.size()); in.read(back.data(),size);
        check(in.grew(),"growth after open not detected");
    }
    std::filesystem::resize_file(file,size-100);
    for(const bool ahead:{false,true}) {
        io::SequentialReader in(file,ahead);
        std::vector<char> back(data.size());
        refused([&]{in.read(back.data(),size);},"truncated stream accepted");
    }
    std::filesystem::remove_all(root);
#else
    static_cast<void>(root);
#endif
}
// Unbuffered publication/restore: same bytes as the buffered path in every mode, old files
// restore, short writes complete, failed writes leave the published checkpoint intact, and
// the written file does not stay in the page cache.
void unbuffered_session_io_tests(const std::filesystem::path& root,const strix::SessionCompatibility& compat) {
    EnvGuard restore_env("STRIX_SESSION_DIRECT_IO");
    auto& faults=strix::state_file_io::fault_injection();
    // Sizes: sub-block, just under one staging chunk, multi-chunk with an unaligned tail.
    for(const std::uint64_t payload:{std::uint64_t{1000},std::uint64_t{(8U<<20)-600},std::uint64_t{(20U<<20)+12345}}) {
        const auto blob=strix::encode_session_checkpoint(large_checkpoint(compat,payload,static_cast<unsigned>(payload%97+1)),heap);
        const auto off=root/"off", on=root/"on";
        set_env("STRIX_SESSION_DIRECT_IO","0"); strix::publish_session_checkpoint(off,blob);
        set_env("STRIX_SESSION_DIRECT_IO","1"); strix::publish_session_checkpoint(on,blob);
        const auto a=file_bytes(off/"session.bin"), b=file_bytes(on/"session.bin");
        check(a.size()==blob.bytes && a==b && !std::memcmp(a.data(),blob.data.get(),a.size()),"unbuffered publication bytes differ");
        EnvGuard restore_readahead("STRIX_SESSION_READAHEAD"), restore_chunked("STRIX_SESSION_CHUNKED");
        for(const char* chunked:{"0","1"}) for(const char* ahead:{"0","1"}) for(const char* mode:{"0","1"}) for(const auto& d:{off,on}) {
            set_env("STRIX_SESSION_CHUNKED",chunked);
            set_env("STRIX_SESSION_READAHEAD",ahead);
            set_env("STRIX_SESSION_DIRECT_IO",mode);
            strix::SessionCheckpoint decoded;
            auto back=strix::read_session_checkpoint(d,compat,heap,blob.bytes,nullptr,&decoded);
            check(back.bytes==blob.bytes && !std::memcmp(back.data.get(),blob.data.get(),static_cast<std::size_t>(blob.bytes)),"restore bytes differ");
            check(decoded.sections.size()==2 && decoded.sections[0].payload.bytes==payload,"restore decode differs");
        }
        set_env("STRIX_SESSION_DIRECT_IO","1");
        refused([&]{(void)strix::read_session_checkpoint(on,compat,heap,blob.bytes-1);},"unbuffered read admission ignored");
        // One flipped byte anywhere (late chunk, footer) is refused by name in every read mode.
        for(const std::uint64_t at:{blob.bytes-blob.bytes/3,blob.bytes-1}) {
            auto bad=a; bad[static_cast<std::size_t>(at)]^=1;
            { std::ofstream out(root/"bad.bin",std::ios::binary); out.write(bad.data(),static_cast<std::streamsize>(bad.size())); }
            std::filesystem::create_directories(root/"bad");
            std::filesystem::rename(root/"bad.bin",root/"bad"/"session.bin");
            for(const char* chunked:{"0","1"}) for(const char* mode:{"0","1"}) {
                set_env("STRIX_SESSION_CHUNKED",chunked); set_env("STRIX_SESSION_DIRECT_IO",mode);
                bool named=false;
                try { (void)strix::read_session_checkpoint(root/"bad",compat,heap,blob.bytes); }
                catch(const std::runtime_error& e) { named=std::string_view(e.what())=="session manifest root mismatch"; }
                check(named,"corrupt envelope not refused by its manifest root");
            }
        }
        set_env("STRIX_SESSION_DIRECT_IO","1"); set_env("STRIX_SESSION_CHUNKED",nullptr);
        std::filesystem::remove_all(root/"bad");
        // Short writes (block-sized and odd-sized) still produce the same file.
        for(const std::uint64_t cap:{std::uint64_t{3*4096},std::uint64_t{1000}}) {
            faults.max_syscall_bytes=cap;
            strix::publish_session_checkpoint(root/"short",blob);
            faults.max_syscall_bytes=0;
            check(file_bytes(root/"short"/"session.bin")==a,"short-write publication differs");
        }
        // A failing write keeps the previous publication and removes its temporary.
        faults.fail_at_offset=blob.bytes/2;
        refused([&]{strix::publish_session_checkpoint(on,blob);},"failed unbuffered write accepted");
        faults.fail_at_offset=UINT64_MAX;
        check(file_bytes(on/"session.bin")==a,"failed write replaced the publication");
        for(const auto& e:std::filesystem::directory_iterator(on))
            check(e.path().filename()=="session.bin","failed write left a temporary");
        // Truncated files are refused on the unbuffered read as on the buffered one.
        std::filesystem::resize_file(on/"session.bin",blob.bytes-7);
        refused([&]{(void)strix::read_session_checkpoint(on,compat,heap,blob.bytes);},"unbuffered truncated file accepted");
        std::filesystem::remove_all(off); std::filesystem::remove_all(on); std::filesystem::remove_all(root/"short");
    }
#if defined(__linux__)
    // Page cache: a resident read-mostly table stays resident across a large save, and the
    // saved file itself is not left in the page cache (direct writes, or the fallback's drops).
    for(const bool refuse:{false,true}) {
        faults.refuse_direct=refuse;
        const auto table=root/"resident-table.bin";
        { std::ofstream t(table,std::ios::binary); std::vector<char> z(1U<<20,'p'); for(int i=0;i<64;++i) t.write(z.data(),static_cast<std::streamsize>(z.size())); }
        { auto bytes=file_bytes(table); check(bytes.size()==(64U<<20),"table size"); }  // make it resident
        const auto before=resident_pages(table);
        const auto blob=strix::encode_session_checkpoint(large_checkpoint(compat,std::uint64_t{256}<<20,7),heap);
        set_env("STRIX_SESSION_DIRECT_IO","1");
        const auto writes=strix::state_file_io::io_counters().direct_writes.load();
        strix::publish_session_checkpoint(root/"big",blob);
        const bool direct=strix::state_file_io::io_counters().direct_writes.load()>writes;
        const auto after=resident_pages(table), saved=resident_pages(root/"big"/"session.bin");
        set_env("STRIX_SESSION_DIRECT_IO","0");
        strix::publish_session_checkpoint(root/"big-off",blob);
        const auto saved_off=resident_pages(root/"big-off"/"session.bin");
        std::cout<<"residency (refuse_direct="<<refuse<<"): table "<<before.first<<"/"<<before.second<<" -> "<<after.first<<"/"<<after.second
                 <<"; saved file direct="<<direct<<" resident "<<saved.first<<"/"<<saved.second
                 <<"; buffered save resident "<<saved_off.first<<"/"<<saved_off.second<<'\n';
        check(after.first>=before.first,"save evicted resident table pages");
        check(direct!=refuse,"fallback selection");
        // EINVAL when O_DIRECT is set on the created file (no direct I/O here): exclusive creation
        // still publishes, buffered, and leaves no temporary behind.
        for(const auto& e:std::filesystem::directory_iterator(root/"big"))
            check(e.path().filename()=="session.bin","fallback publication left a temporary");
        if(page_cache_backed(root)) check(saved.first*100<=saved.second,"unbuffered save left the file in the page cache");
        else std::cout<<"residency: tmpfs, saved-file residency not meaningful\n";
        // Either reader restores the same bytes from a fallback-written file.
        set_env("STRIX_SESSION_DIRECT_IO","1");
        auto back=strix::read_session_checkpoint(root/"big",compat,heap,blob.bytes);
        check(!std::memcmp(back.data.get(),blob.data.get(),static_cast<std::size_t>(blob.bytes)),"fallback restore differs");
        faults.refuse_direct=false;
        std::filesystem::remove(table); std::filesystem::remove_all(root/"big"); std::filesystem::remove_all(root/"big-off");
    }
#endif
}
int main() {
    const auto dir=std::filesystem::temp_directory_path()/
        ("strix-session-tests-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        session_tail_plan_test();
        sequential_reader_tests(dir/"reader");
        old_format_session_tests(dir/"old-format");
        unbuffered_session_io_tests(dir/"unbuffered",strix::SessionCompatibility{hash("full model content"),hash("exe"),"compiled plan;all env",false});
        const auto aligned=strix::plan_session_suffix(32768,1024,1024,true);
        check(aligned.tiled && aligned.serial_rows==0,"aligned suffix did not tile");
        const auto unaligned=strix::plan_session_suffix(32769,2047,1024,true);
        check(unaligned.tiled && unaligned.serial_rows==1023,"unaligned suffix boundary differs");
        check(!strix::plan_session_suffix(32769,2046,1024,true).tiled,"short unaligned suffix tiled");
        check(!strix::plan_session_suffix(32768,128,1024,true).tiled,"short suffix changed route");
        check(!strix::plan_session_suffix(32768,2048,1024,false).tiled,"default suffix route changed");
        strix::SessionCompatibility compat{hash("full model content"),hash("exe"),"compiled plan;all env",false};
        strix::SessionCheckpoint c;
        c.compatibility=compat; c.frontier.consumed=3; c.frontier.accepted=3;
        c.frontier.emitted=2; c.frontier.pending=99; c.frontier.tokens={7,8,9};
        auto payload=heap(4096);
        for(std::uint64_t i=0;i<payload.bytes;++i) payload.data.get()[i]=static_cast<std::byte>(i&255U);
        c.sections.push_back({"target.kv",payload}); c.sections.push_back({"target.fixed",heap(0)});
        auto encoded=strix::encode_session_checkpoint(c,heap);
        {
            const std::array<strix::RotaryIdentity,3> rotary{strix::RotaryIdentity{},
                strix::RotaryIdentity(strix::RotaryDescriptor::yarn(2),hash("table2")),
                strix::RotaryIdentity(strix::RotaryDescriptor::yarn(4),hash("table4"))};
            for(const auto& producer:rotary) {
                auto saved=c; saved.compatibility.numerical_config=strix::session_numerical_config("same CLI",producer);
                auto envelope=strix::encode_session_checkpoint(saved,heap);
                (void)strix::decode_session_checkpoint(envelope,saved.compatibility);
                for(const auto& reader:rotary) if(reader!=producer) {
                    auto expected=saved.compatibility;
                    expected.numerical_config=strix::session_numerical_config("same CLI",reader);
                    bool named=false;
                    try {(void)strix::decode_session_checkpoint(envelope,expected);}
                    catch(const std::runtime_error& e) {named=std::string_view(e.what())=="session compatibility mismatch";}
                    check(named,"cross-rotary session envelope must refuse by name");
                }
            }
        }
        auto decoded=strix::decode_session_checkpoint(encoded,compat);
        check(decoded.frontier.tokens==c.frontier.tokens && decoded.frontier.pending==99U &&
            decoded.frontier.emitted==2 && decoded.frontier.sampling=="greedy","frontier roundtrip");
        check(decoded.sections.size()==2 && decoded.sections[0].name=="target.kv" &&
            decoded.sections[0].payload.bytes==4096 && !std::memcmp(decoded.sections[0].payload.data.get(),payload.data.get(),4096),"sections roundtrip");
        check(strix::encode_session_checkpoint(decoded,heap).bytes==encoded.bytes,"canonical size");
        auto roundtrip=strix::encode_session_checkpoint(decoded,heap);
        check(!std::memcmp(roundtrip.data.get(),encoded.data.get(),static_cast<std::size_t>(encoded.bytes)),"canonical bytes");
        for(int which=0;which<4;++which) {
            auto wrong=compat;
            if(which==0) wrong.model=hash("other weight");
            if(which==1) wrong.executable=hash("other exe");
            if(which==2) wrong.numerical_config+=";flag=changed";
            if(which==3) wrong.mtp=true;
            refused([&]{(void)strix::decode_session_checkpoint(encoded,wrong);},"compat mismatch accepted");
        }
        auto bad=clone(encoded); bad.data.get()[static_cast<std::size_t>(decoded.sections[0].payload.data.get()-encoded.data.get()) + 3]^=std::byte{1};
        refused([&]{(void)strix::decode_session_checkpoint(bad,compat);},"corrupt payload accepted");
        reroot(bad); // manifest valid, section SHA wrong: must still refuse
        refused([&]{(void)strix::decode_session_checkpoint(bad,compat);},"section hash not checked");
        auto schema_bad=clone(encoded); schema_bad.data.get()[8U + std::string_view("strix-session-v1").size()] ^= std::byte{1};
        reroot(schema_bad);
        refused([&]{(void)strix::decode_session_checkpoint(schema_bad,compat);},"wrong schema accepted");
        auto offset_bad=clone(encoded);
        offset_bad.data.get()[static_cast<std::size_t>(decoded.sections[0].payload.data.get()-encoded.data.get()) - 8U] ^= std::byte{1};
        reroot(offset_bad);
        refused([&]{(void)strix::decode_session_checkpoint(offset_bad,compat);},"noncanonical section offset accepted");
        for(const auto n:{0ULL,31ULL,32ULL,100ULL,encoded.bytes-1ULL}) {
            auto short_blob=encoded; short_blob.bytes=n;
            refused([&]{(void)strix::decode_session_checkpoint(short_blob,compat);},"truncation accepted");
        }
        auto invalid=c; invalid.frontier.tokens.pop_back();
        refused([&]{(void)strix::encode_session_checkpoint(invalid,heap);},"history mismatch accepted");
        invalid=c; invalid.frontier.sampling="random";
        refused([&]{(void)strix::encode_session_checkpoint(invalid,heap);},"sampling approximation accepted");
        invalid=c; invalid.frontier.predictor=1;
        refused([&]{(void)strix::encode_session_checkpoint(invalid,heap);},"target-only predictor frontier accepted");
        invalid=c; invalid.sections.push_back(c.sections.front());
        refused([&]{(void)strix::encode_session_checkpoint(invalid,heap);},"duplicate sections accepted");
        strix::PrefixStateCache lookup({encoded.bytes * 2,{},0},heap);
        check(lookup.park_session("good",clone(encoded),2),"park encoded session");
        check(strix::lookup_session_checkpoint(lookup,"good",compat).has_value(),"lookup good");
        auto incompatible = compat; incompatible.numerical_config += " changed";
        check(!strix::lookup_session_checkpoint(lookup,"good",incompatible),"lookup mismatch must miss");
        check(lookup.park_session("bad",clone(bad),2),"park corrupt envelope");
        check(!strix::lookup_session_checkpoint(lookup,"bad",compat),"lookup corruption must miss");
        strix::SessionIoStats io;
        strix::publish_session_checkpoint(dir,encoded,&io);
        check(io.bytes==encoded.bytes,"publish bytes");
        { std::ofstream interrupted(dir/".session-interrupted.tmp",std::ios::binary); interrupted<<"partial"; }
        strix::SessionCheckpoint disk_decoded;
        auto disk=strix::read_session_checkpoint(dir,compat,heap,encoded.bytes,&io,&disk_decoded);
        check(disk_decoded.frontier.pending==99U,"single verification read/decode");
        strix::preflight_session_checkpoint(dir,compat.executable,"compiled plan;",encoded.bytes*4);
        refused([&]{strix::preflight_session_checkpoint(dir,hash("other executable"),"compiled plan;",encoded.bytes*4);},"preflight executable mismatch accepted");
        refused([&]{strix::preflight_session_checkpoint(dir,compat.executable,"other numerical setting",encoded.bytes*4);},"preflight numerical mismatch accepted");
        bool allocated=false;
        const auto tracked=[&](std::uint64_t n) { allocated=true; return heap(n); };
        const auto load=[&](std::uint64_t budget,std::uint64_t available) {
            return strix::read_session_checkpoint(dir,compat,tracked,
                strix::session_checkpoint_envelope_limit(budget,available));
        };
        refused([&]{(void)load(1,UINT64_MAX);},"one-byte restore host budget ignored");
        check(!allocated,"restore allocated before host budget refusal");
        refused([&]{(void)load(encoded.bytes*4,encoded.bytes*4-1);},"restore available memory ignored");
        check(!allocated,"restore allocated before available-memory refusal");
        refused([&]{(void)load(UINT64_MAX,0);},"restore unknown memory accepted");
        check(!allocated,"unknown-memory refusal allocated");
        (void)load(encoded.bytes*4,encoded.bytes*4);
        check(allocated,"admitted restore did not allocate");
        refused([&]{strix::preflight_session_checkpoint(dir,compat.executable,"compiled plan;",1);},"one-byte preflight budget ignored");
        refused([&]{strix::preflight_session_checkpoint(dir,compat.executable,"compiled plan;",encoded.bytes*4-1);},"preflight four-copy budget ignored");
        check(!std::memcmp(disk.data.get(),encoded.data.get(),static_cast<std::size_t>(encoded.bytes)),"atomic publication roundtrip");
        c.frontier.pending=100; auto second=strix::encode_session_checkpoint(c,heap);
#ifndef _WIN32
        strix::publish_session_checkpoint(dir,second); // replace atomically; old inode remains intact
        auto replaced=strix::read_session_checkpoint(dir,compat,heap,second.bytes);
        check(strix::decode_session_checkpoint(replaced,compat).frontier.pending==100U,"atomic replacement");
        std::exception_ptr reader_failure;
        std::jthread reader([&](std::stop_token stop) {
            try {
                while(!stop.stop_requested()) {
                    auto snapshot=strix::read_session_checkpoint(dir,compat,heap,encoded.bytes);
                    const auto pending=strix::decode_session_checkpoint(snapshot,compat).frontier.pending;
                    check(pending==99U || pending==100U,"reader observed an intermediate publication");
                }
            } catch(...) { reader_failure=std::current_exception(); }
        });
        for(unsigned i=0;i<8;++i) strix::publish_session_checkpoint(dir,i%2U?encoded:second);
        reader.request_stop(); reader.join();
        if(reader_failure) std::rethrow_exception(reader_failure);
#endif
        refused([&]{(void)strix::read_session_checkpoint(dir,compat,heap,encoded.bytes-1);},"read admission budget ignored");
        { std::ofstream truncated(dir/"session.bin",std::ios::binary|std::ios::trunc); truncated<<"bad"; }
        refused([&]{(void)strix::read_session_checkpoint(dir,compat,heap,encoded.bytes);},"truncated disk file accepted");
        refused([&]{strix::preflight_session_checkpoint(dir,compat.executable,"compiled plan;",encoded.bytes*4);},"preflight truncation accepted");
        // Same pinned buffer accounting as content prefixes. A borrowed resume handle remains charged after eviction.
        strix::PrefixStateCache lru({32,{},0},heap);
        check(lru.park_session("a",heap(16),2),"park a");
        check(lru.park_session("b",heap(16),2),"park b");
        check(lru.stats().host_bytes==32,"parking byte count");
        { auto held=lru.resume_session("a"); check(held.bytes==16,"resume a"); }
        check(lru.park_session("c",heap(16),2),"park c");
        check(!lru.resume_session("b").data && lru.resume_session("a").data,"count LRU evicted wrong session");
        auto held=lru.resume_session("a");
        check(lru.delete_session("a"),"delete session");
        check(lru.stats().host_bytes==32,"borrowed pinned memory disappeared from budget");
        held={}; check(lru.stats().host_bytes==16,"released handle still charged");
        check(!lru.park_session("too-large",heap(33),2),"oversized session admitted");
        refused([&]{(void)lru.park_session("../escape",heap(1),2);},"path traversal session id accepted");
        check(lru.delete_session("c") && lru.list_sessions().empty() && lru.stats().host_bytes==0,"delete accounting");
        strix::PrefixStateCache bytes_lru({24,{},0},heap);
        check(bytes_lru.park_session("one",heap(16),10) && bytes_lru.park_session("two",heap(16),10),"byte eviction admission");
        check(bytes_lru.list_sessions()==std::vector<std::string>{"two"},"byte budget did not evict oldest");
        const auto model=dir/"model.bin";
        { std::ofstream f(model,std::ios::binary); f<<"first complete file"; }
        auto first=strix::session_cached_file_hash(model);
        check(first==strix::sha256_file_content(model) && first==strix::session_cached_file_hash(model),"full-content cached digest");
        { std::ofstream f(model,std::ios::binary|std::ios::trunc); f<<"second complete file, different length"; }
        check(first!=strix::session_cached_file_hash(model),"file identity cache failed to invalidate");
        const char* old_env = std::getenv("STRIX_SESSION_CPU_TEST");
        const std::optional<std::string> old_value = old_env ? std::optional<std::string>{old_env} : std::nullopt;
        const auto set_test_env = [](const char* value) {
#ifdef _WIN32
            check(_putenv_s("STRIX_SESSION_CPU_TEST",value?value:"")==0,"test environment update failed");
#else
            check((value?setenv("STRIX_SESSION_CPU_TEST",value,1):unsetenv("STRIX_SESSION_CPU_TEST"))==0,"test environment update failed");
#endif
        };
        set_test_env("one"); const auto env_one = strix::session_numerical_config("same effective CLI");
        set_test_env("two"); const auto env_two = strix::session_numerical_config("same effective CLI");
        set_test_env(old_value ? old_value->c_str() : nullptr);
        check(env_one!=env_two,"all-STRIX environment config failed to invalidate");
        check(env_two!=strix::session_numerical_config("different effective CLI"),"effective CLI not bound");
        for (const auto key : {"HIP_SESSION_CPU_TEST", "HSA_SESSION_CPU_TEST", "AMD_SESSION_CPU_TEST", "ROCR_SESSION_CPU_TEST", "GPU_SESSION_CPU_TEST"}) {
            const char* old = std::getenv(key);
            const std::optional<std::string> saved = old ? std::optional<std::string>{old} : std::nullopt;
            const auto set = [&](const char* value) {
#ifdef _WIN32
                check(_putenv_s(key,value?value:"")==0,"runtime environment update failed");
#else
                check((value?setenv(key,value,1):unsetenv(key))==0,"runtime environment update failed");
#endif
            };
            set("one"); const auto one = strix::session_numerical_config("cli");
            set("two"); check(one!=strix::session_numerical_config("cli"),"runtime env not bound");
            set(nullptr); check(!one.starts_with(strix::session_numerical_config("cli")),"removed runtime env accepted as prefix");
            set(saved ? saved->c_str() : nullptr);
        }
        for(const auto* key:{"STRIX_PREFIX_CACHE_INFLIGHT","STRIX_PREFIX_CACHE_DISK","STRIX_PREFIX_CACHE_DURABLE","STRIX_PREFIX_FAIR_RECLAIM_IDLE",
                "STRIX_SESSION_DIRECT_IO","STRIX_SESSION_READAHEAD","STRIX_SESSION_CHUNKED"}) {
            const char* previous=std::getenv(key);
            const std::optional<std::string> saved=previous?std::optional<std::string>{previous}:std::nullopt;
            const auto set=[&](const char* value) {
#ifdef _WIN32
                check(_putenv_s(key,value?value:"")==0,"operational environment update");
#else
                check((value?setenv(key,value,1):unsetenv(key))==0,"operational environment update");
#endif
            };
            set(nullptr);const auto absent=strix::session_numerical_config("same CLI");
            set("0");check(absent==strix::session_numerical_config("same CLI"),"operational off must preserve numerical signature");
            set("1");check(absent==strix::session_numerical_config("same CLI"),"operational on must preserve numerical signature");
            set(saved?saved->c_str():nullptr);
        }
        for (const char* key : {"STRIX_QPF2K_T16K", "STRIX_QPF2K_INDEX_BATCH", "STRIX_EAGER_CODE_LOAD", "HIP_ENABLE_DEFERRED_LOADING", "STRIX_QPF2K_EARLY_SETUP", "STRIX_QPF2K_HC_LOWBAND", "STRIX_QPF2K_COMPACT_IMAGES", "STRIX_QSA_CACHE_ONLY"}) {
            const char* old = std::getenv(key);
            const std::optional<std::string> saved = old ? std::optional<std::string>{old} : std::nullopt;
            const auto set = [&](const char* value) {
#ifdef _WIN32
                check(_putenv_s(key,value?value:"")==0,"startup environment update failed");
#else
                check((value?setenv(key,value,1):unsetenv(key))==0,"startup environment update failed");
#endif
            };
            set(nullptr); const auto before = strix::session_numerical_config("cli");
            for (const char* value : {"0","1"}) {
                set(value);
                check(before == strix::session_numerical_config("cli"),"exact startup setting changed numerical session identity");
            }
            set(saved ? saved->c_str() : nullptr);
        }
        std::filesystem::remove_all(dir);
        std::cout<<"session_state_cache_tests PASS\n"; return 0;
    } catch(const std::exception& e) {
        std::cerr<<"session_state_cache_tests FAIL: "<<e.what()<<'\n';
        std::error_code ec; std::filesystem::remove_all(dir,ec); return 1;
    }
}
