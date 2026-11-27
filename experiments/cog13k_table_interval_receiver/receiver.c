/* Binary64 outward interval receiver; no fast-math or FMA. */
#include <math.h>
#include <float.h>
#include <fenv.h>
#include <stdint.h>
#include <stddef.h>

typedef struct { double l,h; } I;
static double dn(double x){return nextafter(x,-INFINITY);}
static double up(double x){return nextafter(x,INFINITY);}
static I point(double x){return (I){x,x};}
static I add(I a,I b){return (I){dn(a.l+b.l),up(a.h+b.h)};}
static I mul(double k,I a){return k>=0?(I){dn(k*a.l),up(k*a.h)}:(I){dn(k*a.h),up(k*a.l)};}
static I affine(double b,const double *w,const I *x,int n){I v=point(b);for(int i=0;i<n;i++)v=add(v,mul(w[i],x[i]));return v;}
/* Grid h=2^-8. |tanh''| <= 2 => interpolation error <= h*h/4. */
static I tanh_box(double x,const double *t){
    if(x<0){I r=tanh_box(-x,t);return (I){-r.h,-r.l};}
    if(x>=8)return (I){t[4096],1.0};
    double y=x*256; int i=(int)y; double f=y-i;
    /* y scaling is exact for finite x in [0,8]; y-i is exact by Sterbenz,
       including i=0. Bound rounding of 1-f explicitly. */
    I a={t[2*i],t[2*i+1]},b={t[2*i+2],t[2*i+3]};
    double k=1-f;
    I left=mul(k,a),right=mul(f,b),v=add(left,right);
    /* |round(1-f)-(1-f)| <= 2^-53 and |tanh| <= 1.
       Widen for this, the interpolation error, and final rounding. */
    double e=0x1p-18+0x1p-52;
    return (I){fmax(-1,dn(v.l-e)),fmin(1,up(v.h+e))};
}
/* Test/export the endpoint enclosure, including grid boundaries. */
int endpoint(double x,const double *table,double *out){
    if(!isfinite(x)||fegetround()!=FE_TONEAREST||DBL_MANT_DIG!=53||FLT_RADIX!=2)return -1;
    I v=tanh_box(x,table);out[0]=v.l;out[1]=v.h;return 0;
}
/* W 12x32, R 12x12, b/v length 12. Packet envelope and action mask only.
   Return canonical action index (within mask), 255 on abstention, -1 invalid. */
int certify(const double *W,const double *R,const double *b,const double *v,double o,
            const double *center,const double *radius,uint16_t mask,const double *table,
            double *gap,double *ranges){
    if(!mask||!isfinite(o)||fabs(o)>64||fegetround()!=FE_TONEAREST||DBL_MANT_DIG!=53||FLT_RADIX!=2)return -1;
    I p0[12],h[12],next[12],scores[16];int n=0,best=0;
    for(int i=0;i<12;i++){
        if(!isfinite(center[i])||!isfinite(radius[i])||radius[i]<0||fabs(center[i])+radius[i]>64)return -1;
        p0[i]=(I){dn(center[i]-radius[i]),up(center[i]+radius[i])};
    }
    for(int a=0;a<16;a++)if(mask&(1u<<a)){
        I p[12];for(int i=0;i<12;i++){p[i]=add(p0[i],point(W[32*i+a]));h[i]=point(0);}
        for(int j=0;j<3;j++){
            for(int i=0;i<12;i++){I z=add(p[i],affine(b[i],R+12*i,h,12));if(!isfinite(z.l)||!isfinite(z.h)||z.l>z.h)return -1;
                next[i]=(I){tanh_box(z.l,table).l,tanh_box(z.h,table).h};}
            for(int i=0;i<12;i++)h[i]=next[i];
        }
        I q=affine(o,v,h,12);if(!isfinite(q.l)||!isfinite(q.h)||q.l>q.h)return -1;
        scores[n]=q;ranges[2*n]=q.l;ranges[2*n+1]=q.h;
        if(q.l>scores[best].l)best=n;
        n++;
    }
    double g=INFINITY;for(int i=0;i<n;i++)if(i!=best)g=fmin(g,dn(scores[best].l-scores[i].h));
    *gap=g;return g>1e-10?best:255;
}
