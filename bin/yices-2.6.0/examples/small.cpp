/*
 * YICES API: MINIMAL EXAMPLE
 */

#include <assert.h>
#include <stdio.h>
#include <yices.h>

#include <new>
#include <iostream>

/*
 * Throw an exception if Yices runs out-of-memory
 */
static void out_of_mem() {
  std::bad_alloc exception;
  throw exception;
}

int main() {

  printf("Testing Yices %s (%s, %s)\n", yices_version,
	 yices_build_arch, yices_build_mode);

  yices_init();
  yices_set_out_of_mem_callback(out_of_mem);

  int n = 0;
  while (true) {
    n ++;
    try {
      yices_new_context(NULL);
    } catch (std::bad_alloc &ba) {
      std::cerr << "Out of Memory: " << ba.what() << " after " << n << " rounds\n";
      return 1;
    }
  }

  yices_exit();

  return 0;
}
