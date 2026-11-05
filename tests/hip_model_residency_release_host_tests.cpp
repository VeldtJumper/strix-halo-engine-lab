#include "strix/hip_model_residency.hpp"
#include <hip/hip_runtime.h>

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#if defined(__linux__)
#include <sys/wait.h>
#include <unistd.h>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
// Pipe trace: registration (G), live backing (B), drain (D), unregister (U),
// return (R), outer cleanup (W), atexit (X).
void event(char value) {
    if (::write(STDOUT_FILENO, &value, 1) != 1) std::_Exit(90);
}
void at_exit() { event('X'); }

strix::ModelManifest fixture_manifest() {
    strix::ModelManifest manifest;
    manifest.architecture = "residency-fixture";
    strix::ManifestTensor tensor;
    tensor.name = "dense";
    tensor.dimensions = {256};
    tensor.ggml_type = 0;
    tensor.byte_size = 1024;
    tensor.absolute_offset = 4096;
    tensor.role = strix::TensorRole::dense_projection;
    tensor.residency = strix::ResidencyPolicy::resident;
    manifest.total_tensor_bytes = tensor.byte_size;
    manifest.tensors.push_back(std::move(tensor));
    manifest.layout_identity = strix::model_manifest_layout_identity(manifest);
    return manifest;
}

void child(const std::filesystem::path& path, bool fail) {
    require(::setenv("STRIX_MODEL_REGISTER_MLOCK", "0", 1) == 0, "disable page locking in fixture");
    require(::unsetenv("HSA_OVERRIDE_GFX_VERSION") == 0, "use mock device identity");
    require(::setenv("STRIX_STARTUP_MARKERS", "0", 1) == 0, "keep startup diagnostics out of the stderr check");
    require(::setenv("STRIX_HASH_LOG", "0", 1) == 0, "keep hash diagnostics out of the stderr check");
    require(::setenv("STRIX_HASH_CACHE", "0", 1) == 0, "do not write an inherited hash cache");
    require(std::atexit(at_exit) == 0, "register exit observer");
    strix::AllocationGenerationSource generations;
    auto mapped = strix::MappedModelFile::open_read_only(path, "release-fixture", generations);
    const auto manifest = fixture_manifest();
    // This observer outlives residency and detects normal stack unwinding.
    struct Cleanup { ~Cleanup() { event('W'); } } cleanup;
    residency_mock::event = event;
    {
        auto residency = strix::HipModelResidency::create(mapped, manifest, generations);
        require(residency.registered_bytes() != 0, "borrowed residency acquired registration");
        residency_mock::fail_unregister = fail;
        event('B');
    } // Exercise the production destructor and HipModelResidency::release.
    event('R');
}

std::string read_pipe(int fd) {
    std::string result;
    char buffer[256];
    for (;;) {
        const auto bytes = ::read(fd, buffer, sizeof(buffer));
        if (bytes < 0 && errno == EINTR) continue;
        require(bytes >= 0, "read child pipe");
        if (!bytes) break;
        result.append(buffer, static_cast<std::size_t>(bytes));
    }
    ::close(fd);
    return result;
}

bool release_case(const std::filesystem::path& path, bool fail) {
    int events[2], errors[2];
    require(::pipe(events) == 0, "create event pipe");
    require(::pipe(errors) == 0, "create diagnostic pipe");
    const auto pid = ::fork();
    if (pid < 0) {
        const auto fork_error = errno;
        for (const auto fd : {events[0], events[1], errors[0], errors[1]}) ::close(fd);
        if (fork_error == ENOSYS || fork_error == ENOTSUP) return false;
        throw std::runtime_error("fork failed");
    }
    if (!pid) {
        ::close(events[0]); ::close(errors[0]);
        if (::dup2(events[1], STDOUT_FILENO) < 0 || ::dup2(errors[1], STDERR_FILENO) < 0)
            std::_Exit(91);
        ::close(events[1]); ::close(errors[1]);
        try { child(path, fail); }
        catch (const std::exception& error) { std::cerr << error.what() << '\n'; std::_Exit(92); }
        std::exit(0);
    }
    ::close(events[1]); ::close(errors[1]);
    const auto observed = read_pipe(events[0]);
    const auto diagnostic = read_pipe(errors[0]);
    int status = 0;
    pid_t waited;
    do { waited = ::waitpid(pid, &status, 0); } while (waited < 0 && errno == EINTR);
    require(waited == pid && WIFEXITED(status), "child exits normally rather than by signal");
    require(WEXITSTATUS(status) == (fail ? 3 : 0), "borrowed unregister failure exits with status 3");
    require(observed == (fail ? "GBDU" : "GBDURWX"),
        "release drains before unregister; failure skips continuation, stack cleanup and atexit");
    require(diagnostic == (fail ?
        "model residency: unregister failed on borrowed backing; terminating without unmapping\n" : ""),
        "death comes from borrowed-unregister fail-stop branch");
    return true;
}
}
#endif

int main() try {
#if defined(__linux__)
    const auto path = std::filesystem::current_path() /
        ("strix-residency-release-" + std::to_string(::getpid()) + ".bin");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code error; std::filesystem::remove(path, error); }
    } cleanup{path};
    {
        std::ofstream output(path, std::ios::binary);
        const std::string bytes(8192, 'w');
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        require(static_cast<bool>(output), "write mapped fixture");
    }
    if (!release_case(path, false) || !release_case(path, true)) {
        std::cout << "SKIP: fork unavailable\n";
        return 77;
    }
    std::cout << "borrowed residency release host tests passed\n";
    return 0;
#else
    std::cout << "SKIP: borrowed residency death test requires Linux mapping and fork\n";
    return 77;
#endif
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
