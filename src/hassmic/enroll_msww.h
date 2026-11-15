/* Private full-coverage MSWW preparation. Included by enroll.c so capture,
 * cancellation and extraction retain their existing owner/process contract. */
static int msww_float_cmp(const void *a,const void *b)
{ double x=*(const double *)a,y=*(const double *)b;return (x>y)-(x<y); }

/* Same 10ms energy/14dB/350ms merge rule as tools/msww/bursts.py.
 * Measure unpadded audio: padded zeros would invalidate the noise floor.
 * ponytail: one clear utterance; multi-burst phrases need explicit alignment,
 * not a wider MIL mask that admits unrelated audio. */
static int msww_word_window(const char *path,double *start,double *end)
{
    FILE *f=fopen(path,"rb");if(!f)return -1;
    double energy[1200],sorted[1200];int16_t x[160];int n=0,bad=0;
    while(n<1200&&fread(x,2,160,f)==160){double e=0;for(int j=0;j<160;j++)e+=(double)x[j]*x[j];energy[n]=sorted[n]=10*log10(fmax(e/160,1));n++;}
    if(ferror(f)||!feof(f)||n<12)bad=1;
    fclose(f);if(bad)return -1;
    qsort(sorted,n,sizeof *sorted,msww_float_cmp);double at=.2*(n-1);int lo=(int)at;double floor=sorted[lo]+(at-lo)*(sorted[lo+1]-sorted[lo]);
    int a=-1,b=-1,count=0;
    for(int i=0;i<n;){if(energy[i]<=floor+14){i++;continue;}int j=i;
        while(j<n){int on=0;for(int k=j;k<n&&k<j+35;k++)if(energy[k]>floor+14){on=1;break;}if(!on)break;j++;}
        if(j-i>=12){a=i;b=j;count++;}i=j;}
    if(count!=1)return -1;
    *start=fmax(0,a*.01+2-.3);*end=b*.01+2+.5;return 0;
}

static int msww_file_hash(const char *path,char out[65])
{
    FILE *f=fopen(path,"rb");if(!f)return -1;struct sha256 h;sha256_init(&h);unsigned char b[8192],digest[32];size_t n;
    while((n=fread(b,1,sizeof b,f)))sha256_update(&h,b,n);
    int bad=ferror(f);fclose(f);sha256_final(&h,digest);
    for(int k=0;k<32;k++)snprintf(out+2*k,3,"%02x",digest[k]);
    return bad?-1:0;
}
static int msww_string(const struct json *j,int token,char *out,size_t n)
{ return json_type(j,token)==J_STR&&j->t[token].end-j->t[token].start<(int)n&&json_str(j,token,out,n); }
static int msww_basename(const char *s)
{ if(!*s||strstr(s,".."))return 0;for(;*s;s++)if(!isalnum((unsigned char)*s)&&*s!='-'&&*s!='_'&&*s!='.')return 0;return 1; }

/* Bundles are per phrase, model-bound, checksummed, and split by speaker.
 * They contain no personal positive recordings. Missing/other-word bundles
 * preserve legacy enrollment; a matching but corrupt bundle fails closed. */
static int msww_prepare_bundle(const char *dir,const char *model,const char *phrase)
{
    const char *path=env_or("HASSMIC_ENROLL_MSWW_POOL","/data/local/hassmic/enroll-msww-pool/manifest.json");
    FILE *f=fopen(path,"rb");if(!f)return errno==ENOENT?0:-1;char text[32769],root[1024],word[128],want[65],actual[65];
    size_t n=fread(text,1,sizeof text-1,f);int bad=ferror(f)||!feof(f);fclose(f);text[n]=0;
    struct json j={0};if(bad||n==sizeof text-1||strlen(path)>=sizeof root||json_parse(&j,text,n)){json_free(&j);return -1;}
    char normalized_word[128],normalized_phrase[128];
    if(json_int(&j,json_get(&j,0,"version"),0)!=1){json_free(&j);return -1;}
    if(!msww_string(&j,json_get(&j,0,"word"),word,sizeof word)){json_free(&j);return -1;}
    normalized(word,normalized_word,sizeof normalized_word);normalized(phrase,normalized_phrase,sizeof normalized_phrase);
    if(strcmp(normalized_word,normalized_phrase)){json_free(&j);return 0;}
    if(!msww_string(&j,json_get(&j,0,"source_sha256"),want,sizeof want)||strlen(want)!=64||msww_file_hash(model,actual)||strcmp(want,actual)){json_free(&j);return -1;}
    int fit=json_get(&j,0,"fitting_speakers"),cal=json_get(&j,0,"calibration_speakers");
    if(json_type(&j,fit)!=J_ARR||json_type(&j,cal)!=J_ARR||json_len(&j,fit)<1||json_len(&j,fit)>64||json_len(&j,cal)<1||json_len(&j,cal)>64)bad=1;
    if(!bad)JSON_EACH(&j,fit,a){char x[128];if(!msww_string(&j,a,x,sizeof x)||!*x)bad=1;JSON_EACH(&j,cal,b){char y[128];if(!msww_string(&j,b,y,sizeof y)||!*y||!strcmp(x,y))bad=1;}}
    snprintf(root,sizeof root,"%s",path);char *slash=strrchr(root,'/');if(slash)*slash=0;else strcpy(root,".");
    char fitting_digest[65]="";
    for(int k=0;k<2&&!bad;k++){const char *key=k?"calibration_pool":"fitting_pool";int o=json_get(&j,0,key);char name[128],source[1200],dest[1200];
        if(!msww_string(&j,json_get(&j,o,"path"),name,sizeof name)||!msww_basename(name)||!msww_string(&j,json_get(&j,o,"sha256"),want,sizeof want)||strlen(want)!=64){bad=1;break;}
        if(k&&!strcmp(fitting_digest,want)){bad=1;break;}if(!k)snprintf(fitting_digest,sizeof fitting_digest,"%s",want);
        snprintf(source,sizeof source,"%s/%s",root,name);snprintf(dest,sizeof dest,"%s/pool-%s.bin",dir,k?"dev":"train");
        int fd=open(source,O_RDONLY|O_NOFOLLOW);struct stat st;if(fd<0){bad=1;break;}
        if(fstat(fd,&st)||!S_ISREG(st.st_mode)||st.st_size<536||st.st_size>8+100000L*528||(st.st_size-8)%528){close(fd);bad=1;break;}
        unsigned char header[8];uint32_t count;
        if(read(fd,header,8)!=8){close(fd);bad=1;break;}memcpy(&count,header+4,4);
        if(memcmp(header,"MSWP",4)||count<1||count>100000||st.st_size!=8+(long)count*528||lseek(fd,0,SEEK_SET)<0){close(fd);bad=1;break;}
        FILE *in=fdopen(fd,"rb"),*out=fopen(dest,"wx");if(!in||!out){if(in)fclose(in);else close(fd);if(out)fclose(out);bad=1;break;}
        struct sha256 h;sha256_init(&h);unsigned char buf[8192],dg[32];
        while((n=fread(buf,1,sizeof buf,in))){sha256_update(&h,buf,n);if(fwrite(buf,1,n,out)!=n)bad=1;}
        if(ferror(in)||ferror(out))bad=1;
        fclose(in);if(fclose(out))bad=1;sha256_final(&h,dg);
        for(int z=0;z<32;z++)snprintf(actual+2*z,3,"%02x",dg[z]);
        if(strcmp(actual,want))bad=1;
    }
    json_free(&j);return bad?-1:1;
}

static int msww_copy_body(FILE *out,const char *path,double *seconds)
{
    FILE *f=fopen(path,"rb");struct stat st;if(!f)return -1;
    if(fstat(fileno(f),&st)||st.st_size%2||st.st_size<4*RATE*2||st.st_size>16*RATE*2||fseek(f,PAD*2,SEEK_SET)){fclose(f);return -1;}
    long left=st.st_size-4*RATE*2;*seconds=left/(2.*RATE);char buf[8192];int bad=0;
    while(left){size_t n=left<(long)sizeof buf?(size_t)left:sizeof buf;if(fread(buf,1,n,f)!=n||fwrite(buf,1,n,out)!=n){bad=1;break;}left-=n;}
    fclose(f);return bad?-1:0;
}
static int msww_augment_raw(const char *dir,int clip,int prefix,const char *out,double *offset)
{
    FILE *f=fopen(out,"wb");if(!f)return -1;static int16_t zeros[16000];char path[512];int bad=0;double ignored;
    for(int k=0;k<2;k++)if(fwrite(zeros,2,RATE,f)!=RATE)bad=1;
    snprintf(path,sizeof path,"%s/replay-%02d.raw",dir,prefix);if(msww_copy_body(f,path,offset))bad=1;
    if(fwrite(zeros,2,RATE/2,f)!=RATE/2)bad=1;
    snprintf(path,sizeof path,"%s/replay-%02d.raw",dir,clip);if(msww_copy_body(f,path,&ignored))bad=1;
    for(int k=0;k<3;k++)if(fwrite(zeros,2,RATE,f)!=RATE)bad=1;
    if(fclose(f))bad=1;
    *offset+=.5;return bad?-1:0;
}
