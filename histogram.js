import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import Stats from 'three/addons/libs/stats.module.js';

class histogram{
    constructor(hist_size = 1, division = 256){
        this.hist_size = 1;
        this.division = division;
    }

    get_hist(raw_img){
        let r = Array(256).fill(0);
        let g = Array(256).fill(0);
        let b = Array(256).fill(0);
        
        let img = this.getImageData(raw_img);

        img.data.filter( function( value, index, array ) {
            let idx = index%4;
            
            switch( index%4 ) {
                case 0:
                    r[value]++;
                    break;
                case 1:
                    g[value]++;
                    break;
                case 2:
                    b[value]++;
                default:
                    break;
            }
            //インデックス番号を比較して重複データのみ排除
            // return array.indexOf( value ) === index;
        })
        // for(let w=0; w<img.width; w++){
        //     for(let h=0; h<img.height; h++){
        //         let c = this.getPixel(img, h, w);
        //         r[c.r] += 1;
        //         g[c.g] += 1;
        //         b[c.b] += 1;
        //     }
        // }
        // console.log(r)
        const c_max = Math.max(...[...r,...g,...b]);

        r = r.map((val)=> val/c_max);
        g = g.map((val)=> val/c_max);
        b = b.map((val)=> val/c_max);

        return this.createDataStructure(r,g,b);

        // const r_shape = new THREE.Shape();
        // r_shape.moveTo(0,0);
        // for(let i=0; i<this.division; i++){
        //     if(i != 0)
        //         r_shape.lineTo(bar*i, r[i-1]/c_max);

        //     r_shape.lineTo(bar*i, r[i]/c_max);
        // }
        // r_shape.lineTo(1, r[255]/c_max);
        // r_shape.lineTo(1,0);
        // r_shape.lineTo(0,0);

        // let r_geometry = new THREE.ExtrudeGeometry( r_shape, extrudeSettings );
        // let r_material = new THREE.MeshBasicMaterial( { color: 0xFF0000, transparent: true, wireframe: false } );
        // const r_mesh = new THREE.Mesh( r_geometry, r_material ) ;


        // const g_shape = new THREE.Shape();
        // g_shape.moveTo(0,0);
        // for(let i=0; i<this.division; i++){
        //     if(i != 0)
        //         g_shape.lineTo(bar*i, g[i-1]/c_max);

        //     g_shape.lineTo(bar*i, g[i]/c_max);
        // }
        // g_shape.lineTo(1, g[255]/c_max);
        // g_shape.lineTo(1,0);
        // g_shape.lineTo(0,0);
        
        // let g_geometry = new THREE.ExtrudeGeometry( g_shape, extrudeSettings );
        // let g_material = new THREE.MeshBasicMaterial( { color: 0x00FF00, transparent: true, wireframe: false } );
        // const g_mesh = new THREE.Mesh( g_geometry, g_material ) ;
        

        // const b_shape = new THREE.Shape();
        // b_shape.moveTo(0,0);
        // for(let i=0; i<this.division; i++){
        //     if(i != 0)
        //         b_shape.lineTo(bar*i, b[i-1]/c_max);

        //     b_shape.lineTo(bar*i, b[i]/c_max);
        // }
        // b_shape.lineTo(1, b[255]/c_max);
        // b_shape.lineTo(1,0);
        // b_shape.lineTo(0,0);
        
        // let b_geometry = new THREE.ExtrudeGeometry( b_shape, extrudeSettings );
        // let b_material = new THREE.MeshBasicMaterial( { color: 0x0000FF, transparent: true, wireframe: false } );
        // const b_mesh = new THREE.Mesh( b_geometry, b_material ) ;
        
        
        // return [r_mesh, g_mesh, b_mesh];
    }

    getImageData( image ) {
        var canvas = document.createElement( 'canvas' );
        canvas.width = image.videoWidth;
        canvas.height = image.videoHeight;

        var context = canvas.getContext( '2d' );
        context.drawImage( image, 0, 0 );

        return context.getImageData( 0, 0, image.videoWidth, image.videoHeight );
    }

    getPixel( imagedata, x, y) {
        var position = ( x + imagedata.width * y ) * 4, data = imagedata.data;
        return { r: data[ position ], g: data[ position + 1 ], b: data[ position + 2 ] };
    }

    createDataStructure(r, g, b){

        const width = 1;
        const height = 256;
        const size = width * height;

        const data = new Float32Array(4*256);

        for(let i = 0; i < size; i++){
            data[i*4] = r[i];
            data[i*4 + 1] = g[i];
            data[i*4 + 2] = b[i];
            data[i*4 + 3] = 1.0;
        }


        const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat, THREE.FloatType);
        texture.needsUpdate = true;
        return texture;
    }
}


export{histogram};