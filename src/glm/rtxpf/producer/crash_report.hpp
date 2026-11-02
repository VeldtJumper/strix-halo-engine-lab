#pragma once
// Diagnostics: an unhandled structured exception (access violation, ...) prints its code, the faulting
// module + offset and a symbolized stack to stderr, and writes a minidump next to the process
// (STRIX_GLM_RTX_CRASH_DUMP=<path>, default glm_native_service.dmp). Never changes a result.
#include <windows.h>
#include <dbghelp.h>
#include <cstdio>
#include <cstdlib>
#pragma comment(lib, "dbghelp.lib")

namespace strix::glm::rtxpf::producer {
inline void crash_frame(HANDLE proc, DWORD64 pc, unsigned i) {
    char module[MAX_PATH] = "?";
    HMODULE mod = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(pc), &mod))
        GetModuleFileNameA(mod, module, sizeof module);
    alignas(SYMBOL_INFO) char buf[sizeof(SYMBOL_INFO) + 256] = {};
    auto* sym = reinterpret_cast<SYMBOL_INFO*>(buf);
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen = 255;
    DWORD64 disp = 0;
    const bool named = SymFromAddr(proc, pc, &disp, sym) != FALSE;
    IMAGEHLP_LINE64 line{};
    line.SizeOfStruct = sizeof line;
    DWORD ldisp = 0;
    const bool lined = SymGetLineFromAddr64(proc, pc, &ldisp, &line) != FALSE;
    std::fprintf(stderr, "CRASH frame %2u %s+0x%llx %s+0x%llx %s:%lu\n", i, module,
                 static_cast<unsigned long long>(pc - reinterpret_cast<DWORD64>(mod)), named ? sym->Name : "?",
                 static_cast<unsigned long long>(disp), lined ? line.FileName : "?", lined ? line.LineNumber : 0ul);
}
inline LONG WINAPI crash_filter(EXCEPTION_POINTERS* e) {
    HANDLE proc = GetCurrentProcess(), thread = GetCurrentThread();
    std::fprintf(stderr, "CRASH code 0x%08lx at %p (thread %lu)\n", e->ExceptionRecord->ExceptionCode,
                 e->ExceptionRecord->ExceptionAddress, GetCurrentThreadId());
    if (e->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && e->ExceptionRecord->NumberParameters >= 2)
        std::fprintf(stderr, "CRASH access %s address 0x%llx\n", e->ExceptionRecord->ExceptionInformation[0] ? "write" : "read",
                     static_cast<unsigned long long>(e->ExceptionRecord->ExceptionInformation[1]));
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
    SymInitialize(proc, nullptr, TRUE);
    CONTEXT ctx = *e->ContextRecord;
    STACKFRAME64 f{};
    f.AddrPC.Offset = ctx.Rip; f.AddrPC.Mode = AddrModeFlat;
    f.AddrFrame.Offset = ctx.Rbp; f.AddrFrame.Mode = AddrModeFlat;
    f.AddrStack.Offset = ctx.Rsp; f.AddrStack.Mode = AddrModeFlat;
    for (unsigned i = 0; i < 40 && StackWalk64(IMAGE_FILE_MACHINE_AMD64, proc, thread, &f, &ctx, nullptr,
                                                SymFunctionTableAccess64, SymGetModuleBase64, nullptr) && f.AddrPC.Offset; ++i)
        crash_frame(proc, f.AddrPC.Offset, i);
    const char* path = std::getenv("STRIX_GLM_RTX_CRASH_DUMP");
    HANDLE file = CreateFileA(path && *path ? path : "glm_native_service.dmp", GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file != INVALID_HANDLE_VALUE) {
        MINIDUMP_EXCEPTION_INFORMATION mei{GetCurrentThreadId(), e, FALSE};
        MiniDumpWriteDump(proc, GetCurrentProcessId(), file, MiniDumpNormal, &mei, nullptr, nullptr);
        CloseHandle(file);
    }
    std::fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}
inline void install_crash_report() { SetUnhandledExceptionFilter(crash_filter); }
}  // namespace strix::glm::rtxpf::producer
