/* miniz.c — single-TU implementation stub for the vendored miniz.h
 *
 * The header at miniz.h is the single-header amalgamation of richgel999/miniz
 * (3.1.x, public domain). It exposes declarations by default; defining
 * MINIZ_IMPLEMENTATION before #include pulls in the function bodies.
 *
 * This .c file is the ONE translation unit in MccMapStudioDLL.dll where the
 * implementation lives. Other TUs (e.g. MapCacheCommon.cpp) just #include
 * "miniz.h" without defining MINIZ_IMPLEMENTATION and get declarations only.
 *
 * Compiled as C99 (the vcxproj has no PCH attached to this file). The header
 * itself uses #ifdef __cplusplus extern "C" guards so it's binary-compatible
 * either way.
 */

#define MINIZ_IMPLEMENTATION
#include "miniz.h"
