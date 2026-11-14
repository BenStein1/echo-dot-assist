/* Native MSWW enrollment backend. No Python, network, or proprietary assets.
 * msww_train STOCK_ONNX FEATURES.json CANDIDATE.onnx SECONDS
 * FEATURES uses the enrollment array of {path,label,split:train|dev|test}.
 * Five positive/negative recordings fit the row, one of each calibrates its
 * threshold, and two of each are opened only after the row is frozen.
 * msww_row's compressed pool keeps this within the Echo's memory budget.
 */
#define _POSIX_C_SOURCE 200809L
#define main msww_fit_main
#include "msww_row.c"
#undef main
#include "json.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>

#define SLOT 12
#define MAX_CLIPS 64
#define MAX_CYCLES 4096
struct clip { char path[1200]; int label, split; unsigned expected_cycles; };
struct row { int8_t q[D]; float scale, bias; int32_t zero, sums; };
struct blob { unsigned char *p; size_t n; };
struct pb { unsigned id, wire; uint64_t val; unsigned char *p; size_t n; };
static int read_blob(const char *path, struct blob *b, size_t max)
{
    FILE *f=fopen(path,"rb"); if(!f)return -1;
    if(fseek(f,0,SEEK_END)){fclose(f);return -1;}long n=ftell(f);
    if(n<1||(size_t)n>max||fseek(f,0,SEEK_SET)){fclose(f);return -1;}
    b->p=malloc((size_t)n+1); b->n=(size_t)n;
    if(!b->p){fclose(f);return -1;}
    int bad=fread(b->p,1,b->n,f)!=b->n; fclose(f);b->p[b->n]=0; return bad?-1:0;
}
static int var(unsigned char **p,unsigned char *end,uint64_t *v)
{
    *v=0;for(int s=0;s<64;s+=7){if(*p>=end)return -1;unsigned c=*(*p)++;
        if(s==63&&(c&0xfe))return -1;*v|=(uint64_t)(c&127)<<s;if(!(c&128))return 0;}return -1;
}
static int next(unsigned char **p,unsigned char *end,struct pb *f)
{
    uint64_t tag,n;if(*p==end)return 0;if(var(p,end,&tag)||!(tag>>3))return -1;
    memset(f,0,sizeof *f);f->id=tag>>3;f->wire=tag&7;
    if(!f->wire)return var(p,end,&f->val)?-1:1;
    if(f->wire==2){if(var(p,end,&n))return -1;}else if(f->wire==1)n=8;else if(f->wire==5)n=4;else return -1;
    if(n>(uint64_t)(end-*p))return -1;f->p=*p;f->n=n;*p+=n;return 1;
}
struct builder { unsigned char *p;size_t n,cap; };
static int add(struct builder *b,const void *p,size_t n)
{if(n>b->cap-b->n)return -1;memcpy(b->p+b->n,p,n);b->n+=n;return 0;}
static int putvar(struct builder *b,uint64_t v)
{unsigned char p[10];size_t n=0;do{p[n++]=(v&127)|(v>127?128:0);v>>=7;}while(v);return add(b,p,n);}
static int putblob(struct builder *b,unsigned id,const void *p,size_t n)
{return putvar(b,((uint64_t)id<<3)|2)||putvar(b,n)||add(b,p,n);}
static int write_file(const char *path,const void *p,size_t n)
{
    int fd=open(path,O_WRONLY|O_CREAT|O_EXCL,0600);if(fd<0)return -1;int bad=0;
    for(size_t at=0;at<n;){ssize_t w=write(fd,(const char*)p+at,n-at);if(w<=0){bad=1;break;}at+=(size_t)w;}
    if(fsync(fd))bad=1;if(close(fd))bad=1;return bad?-1:0;
}
/* Rebuild the packed sums TensorProto only; fixed-size raw_data of the other
 * four tensors is edited in place. Every competing row and unknown field stays. */
static int save_model(const char *source,const char *dest,const struct row *r)
{
    static const char *names[]={"kernel/Dense:20","kernel/Dense:20_s","kernel/Dense:20_zp","kernel/Dense:20_sums","bias/Dense:20"};
    struct blob m={0};if(read_blob(source,&m,8u<<20))return -1;
    unsigned char *p=m.p,*end=p+m.n,*gb=NULL,*ge=NULL,*graph=NULL;size_t gn=0;struct pb f;int bad=0,found=0;
    while(p<end){unsigned char *begin=p;if(next(&p,end,&f)<1){bad=1;break;}
        if(f.id==7&&f.wire==2){if(graph){bad=1;break;}gb=begin;ge=p;graph=f.p;gn=f.n;}}
    if(bad||!graph){free(m.p);return -1;}
    size_t cap=m.n+4096;unsigned char *gout=malloc(cap),*mout=malloc(cap);
    if(!gout||!mout){free(gout);free(mout);free(m.p);return -1;}
    struct builder g={gout,0,cap},out={mout,0,cap};p=graph;end=graph+gn;
    while(!bad&&p<end){unsigned char *begin=p;if(next(&p,end,&f)<1){bad=1;break;}
        if(f.id!=5||f.wire!=2){bad=add(&g,begin,p-begin);continue;}
        unsigned char *a=f.p,*ae=a+f.n,*raw=NULL;size_t rn=0;char name[128]="";int dtype=0,nd=0,ni=0;uint64_t dims[4];int32_t sums[15];struct pb t;
        while(a<ae){if(next(&a,ae,&t)<1){bad=1;break;}
            if(t.id==8&&t.wire==2){if(t.n>=sizeof name){bad=1;break;}memcpy(name,t.p,t.n);name[t.n]=0;}
            else if(t.id==2&&t.wire==0)dtype=t.val;
            else if(t.id==9&&t.wire==2){raw=t.p;rn=t.n;}
            else if(t.id==1){if(t.wire==0){if(nd==4){bad=1;break;}dims[nd++]=t.val;}
                else if(t.wire==2){unsigned char *q=t.p,*qe=q+t.n;while(q<qe){if(nd==4||var(&q,qe,&dims[nd++])){bad=1;break;}}}}
            else if(t.id==5){if(t.wire==0){if(ni<15)sums[ni]=(int32_t)t.val;ni++;}
                else if(t.wire==2){unsigned char *q=t.p,*qe=q+t.n;while(q<qe){uint64_t v;if(var(&q,qe,&v)){bad=1;break;}if(ni<15)sums[ni]=(int32_t)v;ni++;}}}
        }
        int k;for(k=0;k<5;k++)if(!strcmp(name,names[k]))break;
        if(k==5){if(!bad)bad=add(&g,begin,p-begin);continue;}
        if(found&(1<<k)){bad=1;break;}found|=1<<k;
        if(k==0){if(dtype!=3||nd!=2||dims[0]!=15||dims[1]!=D||rn!=15*D||!raw){bad=1;break;}memcpy(raw+SLOT*D,r->q,D);}
        else {
            int shape=k==4?(nd==2&&dims[0]==1&&dims[1]==15):(nd==1&&dims[0]==15);
            if(!shape||dtype!=((k==1||k==4)?1:6)){bad=1;break;}
            if(k==3){if(raw&&rn==60)memcpy(sums,raw,60);else if(raw||ni!=15){bad=1;break;}
                sums[SLOT]=r->sums;unsigned char buf[2048];struct builder tensor={buf,0,sizeof buf};a=f.p;
                while(a<ae){unsigned char *begin_t=a;if(next(&a,ae,&t)<1){bad=1;break;}
                    if(t.id!=5&&t.id!=9&&add(&tensor,begin_t,a-begin_t)){bad=1;break;}}
                if(!bad)bad=putblob(&tensor,9,sums,60)||putblob(&g,5,tensor.p,tensor.n);continue;
            }
            if(!raw||rn!=60){bad=1;break;}
            memcpy(raw+SLOT*4,k==1?(const void*)&r->scale:k==2?(const void*)&r->zero:(const void*)&r->bias,4);
        }
        if(!bad)bad=add(&g,begin,p-begin);
    }
    if(!bad&&found==31)bad=add(&out,m.p,gb-m.p)||putblob(&out,7,g.p,g.n)||add(&out,ge,m.p+m.n-ge)||write_file(dest,out.p,out.n);
    else bad=1;free(gout);free(mout);free(m.p);return bad?-1:0;
}

struct features { unsigned n;float *data; };
static int load_features(const char *path,struct features *f)
{
    struct blob b={0};if(read_blob(path,&b,16+MAX_CYCLES*528*4))return -1;
    uint32_t h[4];if(b.n<16){free(b.p);return -1;}memcpy(h,b.p,16);
    if(h[0]!=0x46595250||h[1]!=2||h[2]!=512||h[3]!=15||(b.n-16)%(528*4)||b.n==16){free(b.p);return -1;}
    f->n=(b.n-16)/(528*4);f->data=malloc(b.n-16);if(!f->data){free(b.p);return -1;}
    memcpy(f->data,b.p+16,b.n-16);free(b.p);
    for(size_t i=0;i<(size_t)f->n*528;i++)if(!okf(f->data[i])){free(f->data);f->data=NULL;return -1;}return 0;
}
static float competing(const float *y)
{
    double mx=-INFINITY,sum=0;for(int k=0;k<15;k++)if(k!=SLOT&&y[k]>mx)mx=y[k];
    for(int k=0;k<15;k++)if(k!=SLOT)sum+=exp((double)y[k]-mx);return (float)(mx+log(sum));
}
static int safe_name(const char *p)
{if(!*p||strstr(p,".."))return 0;for(;*p;p++)if(!((*p>='a'&&*p<='z')||(*p>='A'&&*p<='Z')||(*p>='0'&&*p<='9')||*p=='_'||*p=='-'||*p=='.'))return 0;return 1;}
static int load_manifest(const char *path,struct clip *c,int *count)
{
    struct blob b={0};struct json j={0};if(read_blob(path,&b,65536)||json_parse(&j,(char*)b.p,b.n)){free(b.p);json_free(&j);return -1;}
    int n=json_len(&j,0),bad=0,i=0,counts[3][2]={{0}};char dir[1024];snprintf(dir,sizeof dir,"%s",path);char *slash=strrchr(dir,'/');if(slash)*slash=0;else strcpy(dir,".");
    if(json_type(&j,0)!=J_ARR||n<16||n>MAX_CLIPS)bad=1;
    if(!bad)JSON_EACH(&j,0,k){char file[200],split[16];
        if(!json_str(&j,json_get(&j,k,"path"),file,sizeof file)||!safe_name(file)||
           !json_str(&j,json_get(&j,k,"split"),split,sizeof split)){bad=1;break;}
        c[i].expected_cycles=0;c[i].label=json_int(&j,json_get(&j,k,"label"),-1);
        c[i].split=!strcmp(split,"train")?0:!strcmp(split,"dev")?1:!strcmp(split,"test")?2:-1;
        if(c[i].label<0||c[i].label>1||c[i].split<0||snprintf(c[i].path,sizeof c[i].path,"%s/%s",dir,file)>=(int)sizeof c[i].path){bad=1;break;}
        for(int k2=0;k2<i;k2++)if(!strcmp(c[k2].path,c[i].path)){bad=1;break;}
        counts[c[i].split][c[i].label]++;i++;
    }
    if(counts[0][0]<5||counts[0][1]<5||!counts[1][0]||!counts[1][1]||counts[2][0]<2||counts[2][1]<2)bad=1;
    *count=i;free(b.p);json_free(&j);return bad?-1:0;
}
static int make_inputs(struct clip *c,int n,double seconds,const char *pool,const char *pos)
{
    FILE *p=fopen(pool,"wb"),*q=fopen(pos,"wb");if(!p||!q){if(p)fclose(p);if(q)fclose(q);return -1;}
    uint32_t pn=0,qn=0;fwrite("MSWP",1,4,p);fwrite(&pn,4,1,p);fwrite("MSWQ",1,4,q);fwrite(&qn,4,1,q);int bad=0;
    for(int k=0;k<n&&!bad;k++)if(c[k].split==0){struct features f={0};if(load_features(c[k].path,&f)){bad=1;break;}
        if(c[k].expected_cycles&&abs((int)f.n-(int)c[k].expected_cycles)>2){free(f.data);bad=1;break;}
        unsigned a=0,b=0;if(c[k].label){while(a<f.n&&(a+1)*.08<1.7)a++;b=a;while(b<f.n&&(b+1)*.08<2+seconds+.2)b++;
            uint32_t nc=b-a;if(!nc){free(f.data);bad=1;break;}fwrite(&nc,4,1,q);qn++;
            for(unsigned i=a;i<b;i++){float *x=f.data+i*528+1,L=competing(x+D);uint32_t mask=(i+1)*.08>=2&&(i+1)*.08<=2+seconds;
                fwrite(&L,4,1,q);fwrite(&mask,4,1,q);fwrite(x,4,D,q);}}
        int first=1;
        for(unsigned i=0;i<f.n;i++){if(c[k].label&&i>=a&&i<b){first=1;continue;}
            float *x=f.data+i*528+1,L=competing(x+D),wt=1,sc=0;int8_t quant[D];uint8_t seq[4]={(uint8_t)(first?1:2),0,0,0};first=0;
            for(int j=0;j<D;j++)if(fabsf(x[j])>sc)sc=fabsf(x[j]);sc=fmaxf(sc/127,1e-8f);
            for(int j=0;j<D;j++)quant[j]=(int8_t)lrintf(x[j]/sc);
            fwrite(&L,4,1,p);fwrite(&wt,4,1,p);fwrite(&sc,4,1,p);fwrite(seq,1,4,p);fwrite(quant,1,D,p);pn++;
        }free(f.data);
    }
    if(!pn||!qn||ferror(p)||ferror(q))bad=1;
    if(fseek(p,4,SEEK_SET)||fseek(q,4,SEEK_SET)||fwrite(&pn,4,1,p)!=1||fwrite(&qn,4,1,q)!=1)bad=1;
    if(fclose(p))bad=1;if(fclose(q))bad=1;return bad?-1:0;
}
static int load_row(const char *path,struct row *r)
{
    struct blob b={0};struct json j={0};if(read_blob(path,&b,65536)||json_parse(&j,(char*)b.p,b.n)){free(b.p);json_free(&j);return -1;}
    int row=json_get(&j,0,"row"),q=json_get(&j,row,"q"),bad=json_len(&j,q)!=D,i=0;
    r->scale=json_num(&j,json_get(&j,row,"s"),NAN);r->bias=json_num(&j,json_get(&j,row,"bias"),NAN);
    r->zero=json_int(&j,json_get(&j,row,"zp"),999);r->sums=json_int(&j,json_get(&j,row,"sums"),INT32_MAX);
    int sum=0;if(!bad)JSON_EACH(&j,q,k){long v=json_int(&j,k,999);if(v< -128||v>127){bad=1;break;}sum+=v;r->q[i++]=(int8_t)v;}
    bad|=!okf(r->scale)||r->scale<=0||!okf(r->bias)||r->zero< -128||r->zero>127||sum!=r->sums;
    free(b.p);json_free(&j);return bad?-1:0;
}
static float row_score(const float *x,const struct row *r)
{
    float lo=x[0],hi=x[0];for(int k=1;k<D;k++){if(x[k]<lo)lo=x[k];if(x[k]>hi)hi=x[k];}
    float sc=(hi-lo)/127.f;if(sc==0)sc=1;float inv=1.f/sc,z=rintf(-lo*inv);int64_t acc=0;
    for(int k=0;k<D;k++){float v=fminf(rintf(x[k]*inv)+z,127.f)-z;acc+=(int64_t)v*((int)r->q[k]-r->zero);}
    return (float)acc*(sc*r->scale)+r->bias;
}
static int scores(const struct clip *c,const struct row *r,float **out,unsigned *n)
{
    struct features f={0};if(load_features(c->path,&f))return -1;
    if(c->expected_cycles&&abs((int)f.n-(int)c->expected_cycles)>2){free(f.data);return -1;}
    float *s=malloc(f.n*4);if(!s){free(f.data);return -1;}
    double sm=0;for(unsigned i=0;i<f.n;i++){const float *x=f.data+i*528+1;double p=sigmoid(row_score(x,r)-competing(x+D));sm=.6*sm+.4*p;s[i]=sm;}
    *out=s;*n=f.n;free(f.data);return 0;
}
static int calibrate(struct clip *c,int n,const struct row *r,double seconds,float *accept)
{
    /* ponytail: the 16-prompt UI affords one dev recording per class, so fit
     * defaults/decay stay fixed and only the threshold is calibrated. More
     * independent recordings would permit the host pipeline's nested search. */
    float pos=1,neg=0;for(int k=0;k<n;k++)if(c[k].split==1){float *s;unsigned nc;if(scores(c+k,r,&s,&nc))return -1;float peak=0;
        for(unsigned i=0;i<nc;i++){double t=(i+1)*.08;if(c[k].label&&t>=1.7&&t<=seconds+2.7){if(s[i]>peak)peak=s[i];}
            else if(s[i]>neg)neg=s[i];}
        if(c[k].label&&peak<pos)pos=peak;free(s);}
    if(!(pos>neg+.02)||pos<=0||neg>=1){fprintf(stderr,"dev recordings do not separate: pos=%g negative=%g\n",pos,neg);return -1;}
    *accept=(pos+neg)/2;return 0;
}
int main(int argc,char **argv)
{
    if(argc!=5){fprintf(stderr,"usage: msww_train SOURCE_ONNX FEATURES.json NEW_ONNX SECONDS\n");return 2;}
    double seconds=atof(argv[4]);if(seconds<2||seconds>6)return 2;
    struct clip clips[MAX_CLIPS];int n=0;if(load_manifest(argv[2],clips,&n)){fprintf(stderr,"invalid MSWW feature manifest\n");return 2;}
    for(int i=0;i<16;i++)clips[i].expected_cycles=(unsigned)((seconds+4)/.08);
    char pool[1400],pos[1400],row[1400],expected[1400];
    if(snprintf(pool,sizeof pool,"%s.pool",argv[3])>=(int)sizeof pool||snprintf(pos,sizeof pos,"%s.pos",argv[3])>=(int)sizeof pos||
       snprintf(row,sizeof row,"%s.row.json",argv[3])>=(int)sizeof row||snprintf(expected,sizeof expected,"%s.replay.json",argv[3])>=(int)sizeof expected)return 2;
    if(make_inputs(clips,n,seconds,pool,pos)){fprintf(stderr,"cannot build MSWW training inputs\n");return 2;}
    pid_t pid=fork();if(pid<0)return 2;if(!pid){char *args[]={"msww_row",pool,pos,row,NULL};_exit(msww_fit_main(4,args));}
    int status;if(waitpid(pid,&status,0)!=pid||!WIFEXITED(status)||WEXITSTATUS(status)){fprintf(stderr,"MSWW row fit failed\n");return 1;}
    struct row r;if(load_row(row,&r))return 1;float accept;
    if(calibrate(clips,n,&r,seconds,&accept))return 1;
    if(save_model(argv[1],argv[3],&r)){fprintf(stderr,"cannot write MSWW decoder\n");return 1;}
    /* The decoder and thresholds are frozen before any test feature is opened. */
    FILE *out=fopen(expected,"wx");if(!out)return 1;
    fprintf(out,"{\"accept\":%.9g,\"notify\":%.9g,\"decay\":0.6,\"recordings\":[",accept,accept/4);
    int counts[2][4]={{0}},bad=0;
    for(int k=0;k<n;k++){float *s;unsigned nc;if(scores(clips+k,&r,&s,&nc)){bad=1;break;}
        int events=0,lock=-1,correct=1;fprintf(out,"%s{\"path\":\"%s\",\"events\":[",k?",":"",strrchr(clips[k].path,'/')+1);
        for(unsigned i=0;i<nc;i++)if((int)(8*(i+1))>lock&&s[i]>=accept){
            unsigned sample=1280*(i+1);fprintf(out,"%s[%u,%.9g]",events?",":"",sample,s[i]);events++;lock=8*(i+1)+100;
            if(sample/16000.0<1.7||sample/16000.0>seconds+2.7)correct=0;
        }fputs("]}",out);free(s);
        int hit=events==1&&correct,outcome=clips[k].label?(hit?0:1):(events?2:3);counts[clips[k].split==2][outcome]++;
        if((clips[k].label&&!hit)||(!clips[k].label&&events))bad=1;
    }
    fputs("]}\n",out);if(ferror(out))bad=1;if(fclose(out))bad=1;
    printf("{\"phase\":\"complete\",\"train\":{\"tp\":%d,\"fn\":%d,\"fp\":%d,\"tn\":%d},\"test\":{\"tp\":%d,\"fn\":%d,\"fp\":%d,\"tn\":%d}}\n",counts[0][0],counts[0][1],counts[0][2],counts[0][3],counts[1][0],counts[1][1],counts[1][2],counts[1][3]);
    return bad?1:0;
}
