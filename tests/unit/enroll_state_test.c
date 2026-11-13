/* Synthetic regression checks: no firmware, model, microphone or recordings needed. */
#include "../../src/hassmic/enroll.c"
#include <assert.h>
#include <sys/file.h>
#ifndef TEST_TMP
#define TEST_TMP "/tmp"
#endif

static void put(const char *name, const char *text)
{
    char p[500]; snprintf(p, sizeof p, "%s/%s", state_dir(), name);
    assert(!statefile_write(p, 0600, text, strlen(text)));
}
static void expect(const char *name, const char *text)
{
    char p[500], got[500] = ""; snprintf(p, sizeof p, "%s/%s", state_dir(), name);
    FILE *f = fopen(p, "r"); assert(f); size_t n = fread(got, 1, sizeof got - 1, f); got[n] = 0; assert(!fclose(f));
    assert(!strcmp(got, text));
}
static atomic_int stop_feed;
static void *feed_thread(void *unused)
{
    (void)unused; int16_t pcm[64] = {0};
    while (!atomic_load(&stop_feed)) enroll_feed(pcm, 64);
    return NULL;
}
int main(int argc, char **argv)
{
    /* Also exercise the same startup/recovery functions against an isolated on-device installer workspace. */
    if (argc == 2 && (!strcmp(argv[1], "--reconcile") || !strcmp(argv[1], "--discard"))) {
        char json[4096], error[300] = ""; enroll_init();
        int rc = !strcmp(argv[1], "--discard") ? enroll_dismiss(error, sizeof error) : 0;
        enroll_json(json, sizeof json); puts(json);
        if (rc) fprintf(stderr, "%s\n", error);
        return rc ? 1 : 0;
    }
    char prompts[8][100];
    for (int i = 0; i < 8; i++) {
        prompt_text("Nova", 0, i, prompts[i], sizeof prompts[i]);
        for (int k = 0; k < i; k++) assert(strcmp(prompts[i], prompts[k]));
    }
    assert(contains_phrase("Say HEY, Computer! now", "Hey Computer"));
    assert(!contains_phrase("Supernova", "Nova"));
    assert(contains_phrase("What time is it?", "time"));
    for (int i = 0; i < 8; i++) { prompt_text("Hey there", 0, i, prompts[i], sizeof prompts[i]); assert(!contains_phrase(prompts[i], "Hey there")); }
    char dir[] = TEST_TMP "/enroll-state-XXXXXX", p[500], err[300] = ""; assert(mkdtemp(dir));
    setenv("HASSMIC_STATE", dir, 1); setenv("HASSMIC_MODELS", dir, 1);
    /* Shared negatives fail closed for incompatible sources, bad splits and traversal. */
    char corpus[500], model[500], data[500];
    snprintf(corpus, sizeof corpus, "%s/corpus.json", dir); setenv("HASSMIC_ENROLL_NEGATIVES", corpus, 1);
    snprintf(model, sizeof model, "%s/base.onnx", dir); put("base.onnx", "abc");
    FILE *manifest = tmpfile(); assert(manifest); int counts[2] = {0};
    assert(!corpus_negatives(manifest, dir, model, "Nova", counts)); /* absent is the documented limited check */
    const char *description = "{\"source_sha256\":\"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\",\"recordings\":[{\"path\":\"train.bin\",\"split\":\"train\",\"text\":\"ordinary speech\",\"speaker\":\"speaker A\"},{\"path\":\"test.bin\",\"split\":\"test\",\"text\":\"different sentence\",\"speaker\":\"speaker B\"}]}";
    put("corpus.json", description);
    const char *bins[] = {"train.bin", "test.bin"};
    for (int i = 0; i < 2; i++) {
        snprintf(data, sizeof data, "%s/%s", dir, bins[i]); FILE *bin = fopen(data, "wb"); assert(bin);
        uint32_t h[] = {0x46595250, 1, 25, 100}; assert(fwrite(h, 1, 16, bin) == 16);
        assert(!fseek(bin, 16 + 31*10000 - 1, SEEK_SET)); assert(fputc(i + 1, bin) == i + 1); assert(!fclose(bin));
    }
    assert(corpus_negatives(manifest, dir, model, "Nova", counts) == 2 && counts[0] == 1 && counts[1] == 1);
    char duplicate[2048]; snprintf(duplicate, sizeof duplicate, "%s", description);
    char *test_text = strstr(duplicate, "different sentence"); assert(test_text);
    memcpy(test_text, "ORDINARY, SPEECH  ", strlen("different sentence"));
    put("corpus.json", duplicate); counts[0] = counts[1] = 0;
    assert(corpus_negatives(manifest, dir, model, "Nova", counts) < 0); /* normalized text overlaps */
    snprintf(duplicate, sizeof duplicate, "%s", description);
    char *test_speaker = strstr(duplicate, "speaker B"); assert(test_speaker); test_speaker[8] = 'A';
    put("corpus.json", duplicate); counts[0] = counts[1] = 0;
    assert(corpus_negatives(manifest, dir, model, "Nova", counts) < 0); /* speaker crosses split */
    put("corpus.json", description);
    snprintf(data, sizeof data, "%s/test.bin", dir); FILE *bin = fopen(data, "r+b"); assert(bin);
    assert(!fseek(bin, -1, SEEK_END)); assert(fputc(1, bin) == 1); assert(!fclose(bin)); counts[0] = counts[1] = 0;
    assert(corpus_negatives(manifest, dir, model, "Nova", counts) < 0); /* same features under different filenames */
    counts[0] = counts[1] = 0;
    assert(corpus_negatives(manifest, dir, model, "ordinary speech", counts) < 0); /* no remaining train split */
    put("base.onnx", "wrong model"); counts[0] = counts[1] = 0;
    assert(corpus_negatives(manifest, dir, model, "Nova", counts) < 0); assert(!fclose(manifest));
    snprintf(p, sizeof p, "%s/installer.lock", dir); setenv("HASSMIC_ARTIFACT_LOCK", p, 1);
    int fence = open(p, O_CREAT | O_RDWR, 0644); assert(fence >= 0);
    snprintf(p, sizeof p, "%s/artifacts", dir); assert(!mkdir(p, 0700));
    snprintf(p, sizeof p, "%s/nova-en-US", dir); assert(!mkdir(p, 0700));
    const char *pending = "nova-en-US\nNova\n1\n0123456789abcdef0123456789abcdef\n";
    put("wake_word", "alexa\n"); put("enroll-pending", pending);
    put("artifacts/result", "OK wake:nova-en-US-other\n");
    assert(!install_result()); expect("wake_word", "alexa\n");
    enroll_init(); assert(S.state == ST_UNRESOLVED && !S.installed); expect("wake_word", "alexa\n");
    put("artifacts/result", "OK wake:nova-en-US\n");
    put("nova-en-US/checksum.txt", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    assert(install_result() && S.state == ST_UNRESOLVED && !S.installed); expect("wake_word", "alexa\n");
    assert(!enroll_dismiss(err, sizeof err) && S.state == ST_IDLE); expect("wake_word", "alexa\n");
    put("enroll-pending", pending);
    put("nova-en-US/checksum.txt", "0123456789abcdef0123456789abcdef");
    enroll_init(); assert(S.state == ST_DONE && S.installed); expect("wake_word", "nova-en-US\n");
    put("wake_word", "alexa\n"); put("enroll-pending", pending);
    put("artifacts/result", "FAILED wake:nova-en-US: simulated refusal\n");
    memset(&S, 0, sizeof S); assert(install_result() && S.state == ST_FAILED); expect("wake_word", "alexa\n");
    /* Permanent silence: a live root lock prevents clearing; once idle, retract only this queued ID. */
    put("enroll-pending", pending); put("artifacts/result", "");
    put("artifacts/request", "wake:nova-en-US\nsound\n");
    put("artifacts/wake.nova-en-US.ready", "ready");
    enroll_init(); assert(S.state == ST_UNRESOLVED);
    assert(!flock(fence, LOCK_EX | LOCK_NB));
    assert(enroll_dismiss(err, sizeof err) == -1 && S.state == ST_UNRESOLVED);
    expect("artifacts/request", "wake:nova-en-US\nsound\n");
    assert(!flock(fence, LOCK_UN));
    assert(!enroll_dismiss(err, sizeof err) && S.state == ST_IDLE);
    expect("artifacts/request", "sound\n");
    snprintf(p, sizeof p, "%s/enroll-pending", dir); assert(access(p, F_OK));
    snprintf(p, sizeof p, "%s/artifacts/wake.nova-en-US.ready", dir); assert(access(p, F_OK));
    /* Confirmed installation with a selection write failure still has an installed model and can recover. */
    snprintf(p, sizeof p, "%s/wake_word", dir); assert(!unlink(p)); assert(!mkdir(p, 0700));
    put("enroll-pending", pending); put("artifacts/result", "OK wake:nova-en-US\n");
    assert(install_result() && S.state == ST_UNRESOLVED && S.installed);
    assert(!enroll_dismiss(err, sizeof err) && S.state == ST_IDLE); assert(!rmdir(p));
    expect("nova-en-US/checksum.txt", "0123456789abcdef0123456789abcdef");
    close(fence);
    for (int state = ST_PACKAGING; state <= ST_DONE; state++) {
        S.state = state; atomic_store(&cancel_flag, 0); enroll_cancel(); assert(!cancelled() && S.state == state);
    }
    S.state = ST_TRAINING; enroll_cancel(); assert(cancelled());
    pthread_t th; atomic_store(&rec_on, 1); assert(!pthread_create(&th, NULL, feed_thread, NULL));
    for (int i = 0; i < 10000; i++) {
        int16_t *b = malloc(128); assert(b);
        pthread_mutex_lock(&mic_lk); rec_buf = b; rec_want = 64; rec_n = 0; pthread_mutex_unlock(&mic_lk);
        rec_release();
    }
    atomic_store(&stop_feed, 1); pthread_join(th, NULL); remove_tree(dir);
    puts("enroll_state: buffer teardown, exact confirmation, refusal, version and restart checks passed");
}
