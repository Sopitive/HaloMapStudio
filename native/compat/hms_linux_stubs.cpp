// =============================================================================
// hms_linux_stubs.cpp - symbols the offline build references but cannot have.
//
// MapCacheUnload.cpp tears down the live-game hooks on cache unload. Those hooks
// live in FramePumpHook.cpp / ForgeBudgetBypass.cpp, which are deliberately NOT
// part of the Linux build (they hook a running MCC, a Windows process). Nothing
// is ever installed here, so removing it is a genuine no-op rather than a
// missing feature. The call sites stay identical across platforms.
// =============================================================================
extern "C" void FramePumpHook_Remove(void)      {}
extern "C" void ForgeBudgetBypass_Remove(void)  {}
