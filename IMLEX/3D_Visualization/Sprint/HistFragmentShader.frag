uniform float time;
varying vec2 vUv;
uniform sampler2D tex;

float getcolor(float thres, float uvy){
    if(thres < uvy)
        return 0.0;
    else
        return 1.0;
}

void main()
{
    vec3 threshold = texture2D(tex, vUv).rgb;
    vec4 color = vec4(1.0,1.0,1.0,1.0);

    if(threshold.x < vUv.x)
        color.x = 0.0;
    if(threshold.y < vUv.x)
        color.y = 0.0;
    if(threshold.z < vUv.x)
        color.z = 0.0;
    
    if(threshold.x < vUv.x && threshold.y < vUv.x && threshold.z < vUv.x)
        discard;

    gl_FragColor = color;

}