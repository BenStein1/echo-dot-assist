/* Package a candidate head using the same configuration policy and checksum as on-device enrollment. */
#include "enroll_pkg.h"
#include <stdio.h>
#include <string.h>
int main(int argc, char **argv)
{
    if (argc != 6) {
        fprintf(stderr, "usage: pryon_pack STOCK_SET CANDIDATE_ONNX OUTPUT_SET KEYWORD NOTE\n");
        return 2;
    }
    char err[300] = "", key[64], id[80];
    if (enroll_name(argv[4], "en-US", key, sizeof key, id, sizeof id, err, sizeof err) || strcmp(key, argv[4])) {
        fprintf(stderr, "keyword must be a normalized custom keyword (letters, digits, underscores): %s\n", err); return 2;
    }
    if (enroll_make_set(argv[1], argv[2], argv[3], argv[4], argv[5], err, sizeof err)) {
        fprintf(stderr, "%s\n", err); return 1;
    }
    return 0;
}
