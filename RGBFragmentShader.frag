varying vec3 color;
mat3 rgb2xyzMat;
uniform int conversion; 


void main() {
    gl_FragColor.rgb = color;
    gl_FragColor.a = 1.0;
}
