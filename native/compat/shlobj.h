#pragma once
#include "hms_windows_shim.h"
#define CSIDL_LOCAL_APPDATA 0x001C
#define CSIDL_APPDATA       0x001A
#define CSIDL_PERSONAL      0x0005
#define S_OK    0
#define E_FAIL  ((long)0x80004005)
typedef long HRESULT;
#define SUCCEEDED(hr) ((HRESULT)(hr) >= 0)
#define FAILED(hr)    ((HRESULT)(hr) <  0)
extern "C" HRESULT SHGetFolderPathW(void* hwnd, int csidl, HANDLE token, DWORD flags, LPWSTR path);
