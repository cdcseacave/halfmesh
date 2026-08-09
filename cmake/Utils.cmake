# cmake/Utils.cmake — halfmesh build helpers

# halfmesh_set_warnings(<target>)
# Applies sensible compiler warning flags to <target> without being
# overly aggressive. Keeps it PRIVATE so downstream consumers are unaffected.
function(halfmesh_set_warnings target)
	if(MSVC)
		target_compile_options(${target} PRIVATE
			/W4       # Warning level 4
			/wd4100   # Suppress: unreferenced formal parameter
			/wd4127   # Suppress: conditional expression is constant
		)
	else()
		# -Wall -Wextra catch the valuable warnings. -Wpedantic is intentionally
		# NOT used: this is a faithful port of code that uses common, well-supported
		# compiler extensions in internal .cc files (e.g. anonymous structs inside
		# unions), which -Wpedantic flags as ISO non-conformance on GCC and Clang
		# alike. Public headers are kept ISO-clean separately (e.g. via __VA_OPT__).
		target_compile_options(${target} PRIVATE
			-Wall
			-Wextra
			-Wno-unused-parameter
		)
	endif()
endfunction()
