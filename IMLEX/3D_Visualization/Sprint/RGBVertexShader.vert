uniform sampler2D tex;
varying vec3 color;

mat3 rgb2xyzMat;
uniform int conversion; 

float rgb2linear(float value){
    return (value <= 0.04045)? value/12.92 : pow((value+0.055)/1.055, 2.4);
}

float f(float value){
    return (value > 0.008856452)? pow(value, 1.0/3.0) : value/3.0/0.042806183+0.137931034;
}

vec3 XYZ2XYY(vec3 xyz){
    vec3 xyy;
    xyy.x = xyz.x/(xyz.x+xyz.y+xyz.z);
    xyy.z = xyz.y/(xyz.x+xyz.y+xyz.z);
    xyy.y = 1.0-xyy.x-xyy.y;
    return xyy;
}

vec3 XYZ2LAB(vec3 xyz){
    vec3 lab;
    lab.y = 116.0*f(xyz.y/100.0)-16.0;
    lab.x = 500.0*(f(xyz.x/95.0489)-f(xyz.y/100.0));
    lab.z = 200.0*(f(xyz.y/100.0)-f(xyz.z/108.884));
    
    return lab;
}

vec3 sRGB2XYY(vec3 value){
        
    vec3 pixel = vec3(rgb2linear(value.x), rgb2linear(value.y), rgb2linear(value.z));
    return XYZ2XYY( rgb2xyzMat*pixel);
}

vec3 sRGB2LAB(vec3 value){
    vec3 rgb_linear = vec3(rgb2linear(value.x), rgb2linear(value.y), rgb2linear(value.z));
    return XYZ2LAB(rgb2xyzMat*rgb_linear);
}

void main() {

    color = texture2D ( tex, position.xy ).rgb;

    rgb2xyzMat[0] = vec3(0.4124, 0.3576, 0.1805);
    rgb2xyzMat[1] = vec3(0.2126, 0.7152, 0.0722);
    rgb2xyzMat[2] = vec3(0.0193, 0.1192, 0.9505);
    
    if(conversion == 1){//XYY
        color.rgb = sRGB2XYY(color);
    }
    else if(conversion == 2){//LAB
        color.rgb = sRGB2LAB(color);
    }
    else{// RGB
        color.rgb = color;
    }

    gl_PointSize = 1.0;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(color, 1.0);
}