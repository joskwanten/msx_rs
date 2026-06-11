/* Golden-master harness: drives the reference C YM2413 with the exact
   register script from src/ym2413.rs's tests and prints an FNV-1a hash
   of 50000 mono samples. Compare against the Rust test's printed hash.
   Build: cc -O2 -I tools -o /tmp/ym2413_ref tools/ym2413_ref.c -lm     */
#include <stdio.h>
#include "/Users/j.j.kwanten/Projects/Msx/RogueDrive/Nextor/ym2413.c"

static void chip(int r, int v) { YM2413Write(0, r); YM2413Write(1, v); }

int main(void) {
  YM2413Init();
  YM2413ResetChip();
  YM2413Write(2, 1); /* SMS output latch ON — Rust forces this in reset() */

  int seq[][2] = {
    {0,0x61},{1,0x61},{2,0x1E},{3,0x17},{4,0xF0},{5,0x7F},{6,0x00},{7,0x17},
    {0x30,0x10},{0x10,0xAB},{0x20,0x18},
    {0x31,0x03},{0x11,0x55},{0x21,0x3A},
    {0x32,0x37},{0x12,0xCD},{0x22,0x12},
    {0x16,0x20},{0x26,0x05},{0x17,0x50},{0x27,0x05},{0x18,0xC0},{0x28,0x01},
    {0x0E,0x3F},
  };
  for (unsigned i = 0; i < sizeof(seq)/sizeof(seq[0]); i++)
    chip(seq[i][0], seq[i][1]);

  unsigned long long hash = 0xcbf29ce484222325ULL;
  int buf[2];
  for (int i = 0; i < 50000; i++) {
    YM2413Update(buf, 1);
    hash ^= (unsigned long long)(long long)buf[0];
    hash *= 0x100000001b3ULL;
  }
  printf("%016llx\n", hash);
  return 0;
}
