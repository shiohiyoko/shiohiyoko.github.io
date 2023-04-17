mat3 rgb2xyzMat;
rgb2xyzMat[0] = vec3(0.4124, 0.3576, 0.1805);
rgb2xyzMat[1] = vec3(0.2126, 0.7152, 0.0722);
rgb2xyzMat[2] = vec3(0.0193, 0.1192, 0.9505);

float rgb2linear(float value){
    return (value <= 0.04045)? value/12.92 : pow((value+0.055)/1.055, 2.4);
}

float f(float value){
    return (value > 0.008856452)? pow(value, 1.0/3.0) : value/3.0/0.042806183+0.137931034;
}

float sRGB2XYZ(vec3 value){
        
    vec3 pixel = (rgb2linear(value.x)), rgb2linear(value.y), rgb2linear(value.z));
    return rgb2xyzMat*pixel;
}

function sRGB2LAB(vec3 value){
    vec3 pixel = (rgb2linear(value.x), rgb2linear(value.y), rgb2linear(value.z));
    return rgb2xyzMat*pixel;
}


float XYZ2XYY(vec3 xyz){
    vec3 xyy
    xyy.x = xyz.x/(xyz.x+xyz.y+xyz.z);
    xyy.y = xyz.y/(xyz.x+xyz.y+xyz.z);
    xyy.z = 1-xyz.x-xyz.y;
    return xyy;
}

float XYZ2LAB(vec3 xyz){
    vec3 lab;
    lab.x = 116.0*f(xyz.y/100.0)-16.0;
    lab.y = 500.0*(f(xyz.x/95.0489)-f(xyz.y/100.0));
    lab.z = 200.0*(f(xyz.y/100.0)-f(xyz.z/108.884));
    
    return lab;
}
