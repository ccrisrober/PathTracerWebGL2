/** @type {Array} */
var _0x398e = [",", "split", "webgl2,experimental-webgl2,webgl,experimental-webgl", "length", "getContext", "ms,moz,webkit,o", "requestAnimationFrame", "RequestAnimationFrame", "cancelAnimationFrame", "CancelAnimationFrame", "CancelRequestAnimationFrame", "0", "#", "resetStaticID", "spheres", "boxes", "triangles", "meshes", "planes", "scenes/", ".xml", "light", "value", "lightPositionX", "getElementById", "center", "lightPositionY", "lightPositionZ", "cam", "lightColor", "getColor", "", "log", "concat", 
"lights", "Compiling shader...", "Compilando", "resetShader", "Oki", "fragmentSource", "stepProgram", "reset", "deleteTexture", "Loading Texture...", "asyncLoadHDRTexture", "fog", "click", "addEventListener", "exposure", "brightnessSlider", "input", "brightness", "dofSpreadSlider", "dofSpread", "focalLengthSlider", "focalLength", "refresh", "replace", "exec", "setColor", "lightIntensity", "emission", "material", "setIntensity", "ior", "defaultIOR", "updIOR", "Update IOR ...", "downloadShader", "text/plain", 
"createObjectURL", "URL", "a", "createElement", "download", "shader.frag", "href", "Reset ...", "bounces", "Update bounces ...", "numSamples", "MAX_SAMPLES", "noneBox", "display", "style", "none", "textureQuadSimpleProgram", "resetRenderProgram", "linearBox", "block", "textureQuadGammaProgram", "reinhardBox", "textureQuadReinhardProgram", "filmicBox", "textureQuadFilmicProgram", "sRGBBox", "textureQuadsRGBProgram", "uncharted2Box", "textureQuadUncharted2Program", "envConstantBox", "accumulation += attenuation * 1.5;\n", 
"environmentRotationSlider", "rotation", "environmentIntensitySlider", "mult", "maxTimeSlider", "shutterAppertureSlider", "../images/hdrTest1.hdr", "2", "hdrTest1", "../images/hdrTest2.hdr", "0.5", "hdrTest2", "../images/hdrTest3.hdr", "1.2", "hdrTest3", "sceneSelector", "change", "Loading scene...", "renderMode", "Ray tracer", "Nothing to do here...", "\n              ........\n             . o   o  .\n            .  ______  .\n             .        .\n              ........\n              .........\n             .    .  .\n            . ..  . . ###\n             .. ....   #####\n             .  .       #######\n            .  . \n          ", 
"Path tracer", "Spherical Armonics", "sf", "getAttribute", "item", "selectedOptions", "Loading shader", "status", "innerHTML", "parentNode", "accumulation += + attenuation * 1.5;", "keydown", "keyCode", "preventDefault", "fromCharCode", "processKeyboard", "W", "S", "A", "D", "E", "Q", "processMouseMovement", "now", "timeElapsed", "update", "step", "render", "getInstance", "nextTextureID", "canvas", "width", "height", "webgl-impl", "Could not load WebGL Context", "WEBGL not supported", "ERROR", "OES_texture_float", 
"getExtension", "OES_texture_float_linear", "Please, update your navegator ...", ".", "indexOf", "e", ".0", "vec2(", ")", "vec3(", "epsilon", "smallEpsilon", "montecarloActive", "vertexCode", "#version 300 es\nin vec3 vertex;\nout vec2 texCoord;\nvoid main() {\n  texCoord = vertex.xy * 0.5 + 0.5;\n  gl_Position = vec4( vertex, 1 );\n}", "toFloat", "toVec2", "toVec3", "id", "globalSourceFragIdCounter", "src", "dependencies", "toString", "prototype", "Dependency not found", "PI", "#define PI 3.14159\n", 
"TWO_PI", "#define TWO_PI 6.2831\n", "rand", "\nhighp float rand(vec2 co) {\n  highp float a = 12.9898;\n  highp float b = 78.233;\n  highp float c = 43758.5453;\n  highp float dt= dot(co.xy ,vec2(a,b));\n  highp float sn= mod(dt,3.1415926);\n  return fract(sin(sn) * c);\n}\n", "pseudorandom", "\nfloat pseudorandom(float u) {\n  float a = fract(sin(gl_FragCoord.x*12.9898*3758.5453));\n  float b = fract(sin(gl_FragCoord.x*63.7264*3758.5453));\n  return rand(gl_FragCoord.xy * mod(u * 4.5453,3.1415926));\n}\n", 
"rayT", "\nvec3 rayT( vec3 rayPos, vec3 rayDir, float t ) {\n  return rayPos + t * rayDir;\n}\n", "rayIntersectPlane", "float rayIntersectPlane( vec3 normal, float d, vec3 rayPos, vec3 rayDir ) {\n  return ( d - dot( normal, rayPos ) ) / dot( normal, rayDir );\n}\n", "rayIntersectBox", "vec2 rayIntersectBox( vec3 boxMinCorner, vec3 boxMaxCorner, vec3 rayPos, vec3 rayDir ) {\n  vec3 tBack = ( boxMinCorner - rayPos ) / rayDir;\n  vec3 tFront = ( boxMaxCorner - rayPos ) / rayDir;\n  vec3 tMin = min( tBack, tFront );\n  vec3 tMax = max( tBack, tFront );\n  float tNear = max( max( tMin.x, tMin.y ), tMin.z );\n  float tFar = min( min( tMax.x, tMax.y ), tMax.z );\n  return vec2( tNear, tFar );\n}\n", 
"normalForBox", "\nvec3 normalForBox( vec3 boxCenter, vec3 boxHalfSize, vec3 point ) {\n  vec3 unitDelta = ( point - boxCenter ) / boxHalfSize;\n  return normalize( step( 1.0 - ", ", unitDelta ) - step( 1.0 - ", ", -1.0 * unitDelta ) );\n}\n", "rayIntersectSphere", "\nvec2 rayIntersectSphere( vec3 center, float radius, vec3 rayPos, vec3 rayDir ) {\n  vec3 toSphere = rayPos - center;\n  float a = dot( rayDir, rayDir );\n  float b = 2.0 * dot( toSphere, rayDir );\n  float c = dot( toSphere, toSphere ) - radius * radius;\n  float discriminant = b * b - 4.0 * a * c;\n  if( discriminant > ", 
" ) {\n    float sqt = sqrt( discriminant );\n    return ( vec2( -sqt, sqt ) - b ) / ( 2.0 * a );\n  } else {\n    return vec2( 1.0, -1.0 );\n  }\n}\n", "normalForSphere", "\nvec3 normalForSphere( vec3 center, float radius, vec3 point ) {\n  return ( point - center ) / radius;\n}\n", "rayIntersectTriangle", "float intersectTriangle(vec3 origin, vec3 dir, vec3 v0, vec3 v1, vec3 v2) {\n\n  vec3 e1 = v1-v0;\n  vec3 e2 = v2-v0;\n  vec3 tvec = origin - v0;  \n\n  vec3 pvec = cross(dir, e2);  \n  float det  = dot(e1, pvec);   \n\n  // check face\n  //if(det < ", 
") return INFINITY;\n\n  float inv_det = 1.0/ det;  \n\n  float u = dot(tvec, pvec) * inv_det;  \n\n  if (u < 0.0 || u > 1.0) return INFINITY;  \n\n  vec3 qvec = cross(tvec, e1);  \n\n  float v = dot(dir, qvec) * inv_det;  \n\n  if (v < 0.0 || (u + v) > 1.0) return INFINITY;  \n\n  float t = dot(e2, qvec) * inv_det;\nreturn t;\n}\n", "triangleNormal", "vec3 normalForTriangle( vec3 v0, vec3 v1, vec3 v2 ) {\n  vec3 e1 = v1-v0;\n  vec3 e2 = v2-v0;\n  return normalize(cross(e2, e1));\n}\n", "rayIntersectMesh", 
"\nvec2 intersectMesh( vec3 origin, vec3 dir, vec3 aabb_min, vec3 aabb_max) {\n  \n    //for(int index = 0; index < 16; index++) {\n    int index = 0;\n    ivec3 list_pos = //texture(triangles_list, vec2((float(index)+0.5)/float(36.0), 0.5)).xyz;\n    //list_pos = vec3(0, 1, 2);\n    if((index+1) % 2 !=0 ) { \n      list_pos.xyz = list_pos.zxy;\n    }\n    vec3 v0 = texture(vertex_positions, vec2((float(list_pos.z) + 0.5 )/float(28.0), 0.5)).xyz;\n    vec3 v1 = texture(vertex_positions, vec2((float(list_pos.y) + 0.5 )/float(28.0), 0.5)).xyz;\n    vec3 v2 = texture(vertex_positions, vec2((float(list_pos.x) + 0.5 )/float(28.0), 0.5)).xyz;\n\n    float hit = intersectTriangle(origin, dir, v0, v1, v2);\n    if(hit < t) { \n      t = hit;\n      tri = index;\n    }\n    //}\n  return vec2(t, float(tri/3));\n}\n", 
"normalOnMesh", "vec3 normalOnMesh( float tri ) {\n  int index = int(tri);\n\n  ivec3 list_pos = indices[index];    //texture(triangles_list, vec2((float(index)+0.5)/float(TRIANGLE_TEX_SIZE), 0.5)).xyz;\n  if((index+1) % 2 !=0 ) { \n    list_pos.xyz = list_pos.zxy;\n  }\n  vec3 v0 = texture(vertex_positions, vec2((float(list_pos.z) + 0.5 )/float(VERTEX_TEX_SIZE), 0.5)).xyz * 30.0 + vec3(0, 75, 0);\n  vec3 v1 = texture(vertex_positions, vec2((float(list_pos.y) + 0.5 )/float(VERTEX_TEX_SIZE), 0.5)).xyz * 30.0 + vec3(0, 75, 0);\n  vec3 v2 = texture(vertex_positions, vec2((float(list_pos.x) + 0.5 )/float(VERTEX_TEX_SIZE), 0.5)).xyz * 30.0 + vec3(0, 75, 0);\n\n  return normalForTriangle( v0, v1, v2 );\n}\n", 
"uniformInsideDisk", "\nvec2 uniformInsideDisk( float xi1, float xi2 ) {\n  float angle = TWO_PI * xi1;\n  float mag = sqrt( xi2 );\n  return vec2( mag * cos( angle ), mag * sin( angle ) );\n}\n", "sampleUniformOnSphere", "vec3 sampleUniformOnSphere( float xi1, float xi2 ) {\n  float angle = TWO_PI * xi1;\n  float mag = 2.0 * sqrt( xi2 * ( 1.0 - xi2 ) );\n  return vec3( mag * cos( angle ), mag * sin( angle ), 1.0 - 2.0 * xi2 );\n}\n", "sampleUniformOnHemisphere", "\nvec3 sampleUniformOnHemisphere( float xi1, float xi2 ) {\n  return sampleUniformOnSphere( xi1, xi2 / 2.0 );\n}\n", 
"sampleDotWeightOnHemiphere", "\nvec3 sampleDotWeightOnHemiphere( float xi1, float xi2 ) {\n  float angle = TWO_PI * xi1; /* Polar angle PHI */\n  float mag = sqrt( xi2 ); /* sin(Polar angle THETA) */\n  /* cos(THETA) = (1-r2)^(1/e+1); We use e=1; */\n  /* p = sin(THETA)cos(PHI)*u + sin(THETA)sin(PHI)*v + cos(THETA)*w; */ \n  return vec3( mag * cos( angle ), mag * sin( angle ), sqrt( 1.0 - xi2 ) ); \n}\n", "samplePowerDotWeightOnHemiphere", "\nvec3 samplePowerDotWeightOnHemiphere( float n, float xi1, float xi2 ) {\n  float angle = TWO_PI * xi1;\n  float z = pow( abs( xi2 ), 1.0 / ( n + 1.0 ) );\n  float mag = sqrt( 1.0 - z * z );\n  return vec3( mag * cos( angle ), mag * sin( angle ), z );\n}\n", 
"sampleDotWeightOn3Hemiphere", "\nvec4 sampleDotWeightOnHemiphere( float xi1, float xi2, float xi3 ) {\n  float tr = pow( abs( xi1 ), 1.0/3.0 );\n  float mag = tr * sqrt( xi2 * ( 1 - xi2 ) );\n  float angle = TWO_PI * xi3;\n  return vec4( mag * cos( angle ), mag * sin( angle ), tr * ( 1 - 2 * xi2 ), sqrt( 1 - tr * tr ) );\n}\n", "sampleTowardsNormal", "\nvec3 sampleTowardsNormal( vec3 normal, vec3 sampleDir ) {\n  vec3 a, b;\n  if ( abs( normal.x ) < 0.5 ) {\n    a = normalize( cross( normal, vec3( 1, 0, 0 ) ) );\n  } else {\n    a = normalize( cross( normal, vec3( 0, 1, 0 ) ) );\n  }\n  b = normalize( cross( normal, a ) );\n  return a * sampleDir.x + b * sampleDir.y + normal * sampleDir.z;\n}\n", 
"totalInternalReflectionCutoff", "\nfloat totalInternalReflectionCutoff( float na, float nb ) {\n  if ( na <= nb ) {\n    return 0.0;\n  }\n  float ratio = nb / na;\n  return sqrt( 1.0 - ratio * ratio );\n}\n", "fresnelDielectric", "vec2 fresnelDielectric( vec3 incident, vec3 normal, vec3 transmitted, float na, float nb ) {\n  float doti = abs( dot( incident, normal ) );\n  float dott = abs( dot( transmitted, normal ) );\n  vec2 result = vec2( ( na * doti - nb * dott ) / ( na * doti + nb * dott ), ( na * dott - nb * doti ) / ( na * dott + nb * doti ) );\n  return result * result;\n}\n", 
"fresnel", "\nvec2 fresnel( vec3 incident, vec3 normal, float na, float nb, float k ) {\n  float doti = abs( dot( incident, normal ) );\n  float comm = na * na * ( doti * doti - 1.0 ) / ( ( nb * nb + k * k ) * ( nb * nb + k * k ) );\n  float resq = 1.0 + comm * ( nb * nb - k * k );\n  float imsq = 2.0 * comm * nb * k;\n  float temdott = sqrt( resq * resq + imsq * imsq );\n  float redott = ( sqrt( 2.0 ) / 2.0 ) * sqrt( temdott + resq );\n  float imdott = ( imsq >= 0.0 ? 1.0 : -1.0 ) * ( sqrt( 2.0 ) / 2.0 ) * sqrt( temdott - resq );\n  float renpdott = nb * redott + k * imdott;\n  float imnpdott = nb * imdott - k * redott;\n  float retop = na * doti - renpdott;\n  float rebot = na * doti + renpdott;\n  float retdet = rebot * rebot + imnpdott * imnpdott;\n  float reret = ( retop * rebot + -imnpdott * imnpdott ) / retdet;\n  float imret = ( -imnpdott * rebot - retop * imnpdott ) / retdet;\n  float sReflect = reret * reret + imret * imret;\n  retop = ( nb * nb - k * k ) * doti - na * renpdott;\n  rebot = ( nb * nb - k * k ) * doti + na * renpdott;\n  float imtop = -2.0 * nb * k * doti - na * imnpdott;\n  float imbot = -2.0 * nb * k * doti + na * imnpdott;\n  retdet = rebot * rebot + imbot * imbot;\n  reret = ( retop * rebot + imtop * imbot ) / retdet;\n  imret = ( imtop * rebot - retop * imbot ) / retdet;\n  float pReflect = reret * reret + imret * imret;\n  return vec2( sReflect, pReflect );\n}\n", 
"sampleFresnelDielectric", "vec3 sampleFresnelDielectric( vec3 incident, vec3 normal, float na, float nb, float xi1 ) {\n  vec3 transmitDir = refract( incident, normal, na / nb );\n  vec2 reflectance = fresnelDielectric( incident, normal, transmitDir, na, nb );\n  if ( xi1 > ( reflectance.x + reflectance.y ) / 2.0 ) {\n    return transmitDir;\n  } else {\n    return reflect( incident, normal );\n  }\n}\n", "SHCoefficients", "struct SHCoefficients {\n    vec3 l00, l1m1, l10, l11, l2m2, l2m1, l20, l21, l22;\n};\n", 
"SHCoeffsGrace", "\nconst SHCoefficients sph_arm = SHCoefficients(\n  vec3( 1.630401,  1.197034,  1.113651),\n  vec3( 0.699675,  0.540300,  0.536132),\n  vec3(-0.354008, -0.287976, -0.268514),\n  vec3( 1.120136,  0.854082,  0.880019),\n  vec3( 1.012764,  0.783031,  0.812029),\n  vec3(-0.181137, -0.147510, -0.132195),\n  vec3(-0.589437, -0.434048, -0.452781),\n  vec3(-0.266943, -0.211540, -0.210316),\n  vec3( 0.868657,  0.665028,  0.655598)\n);\n", "SHCoeffsGalileo", "\nconst SHCoefficients sph_arm = SHCoefficients(\n  vec3( 0.7953949,  0.4405923,  0.5459412 ),\n  vec3( 0.3981450,  0.3526911,  0.6097158 ),\n  vec3(-0.3424573, -0.1838151, -0.2715583 ),\n  vec3(-0.2944621, -0.0560606,  0.0095193 ),\n  vec3(-0.1123051, -0.0513088, -0.1232869 ),\n  vec3(-0.2645007, -0.2257996, -0.4785847 ),\n  vec3(-0.1569444, -0.0954703, -0.1485053 ),\n  vec3( 0.5646247,  0.2161586,  0.1402643 ),\n  vec3( 0.2137442, -0.0547578, -0.3061700 )\n);\n", 
"SHCoeffsBeach", "\nconst SHCoefficients sph_arm = SHCoefficients(\n  vec3( 2.479083,  2.954692,  3.087378),\n  vec3( 1.378513,  1.757425,  2.212955),\n  vec3(-0.321538, -0.574806, -0.866179),\n  vec3( 1.431262,  1.181306,  0.620145),\n  vec3( 0.580104,  0.439953,  0.154851),\n  vec3(-0.446477, -0.688690, -0.986783),\n  vec3(-1.225432, -1.270607, -1.146588),\n  vec3( 0.274751,  0.234544,  0.111212),\n  vec3( 2.098766,  2.112738,  1.652628)\n);\n", "SHCoeffsNeighbor", "\nconst SHCoefficients sph_arm = SHCoefficients(\n  vec3( 2.283449,  2.316459,  2.385942),\n  vec3(-0.419491, -0.409525, -0.400615),\n  vec3(-0.013020, -0.004712,  0.007341),\n  vec3( 0.050598,  0.052119,  0.052227),\n  vec3(-0.319785, -0.315880, -0.312267),\n  vec3( 0.700243,  0.703244,  0.718901),\n  vec3(-0.086157, -0.082425, -0.073467),\n  vec3(-0.242395, -0.228124, -0.218358),\n  vec3( 0.001888,  0.012843,  0.029843)\n);\n", 
"calcIrradiance", "vec3 calcIrradiance(vec3 nor) {\n    const float c1 = 0.429043;\n    const float c2 = 0.511664;\n    const float c3 = 0.743125;\n    const float c4 = 0.886227;\n    const float c5 = 0.247708;\n    return (\n        c1 * sph_arm.l22 * (nor.x * nor.x - nor.y * nor.y) +\n        c3 * sph_arm.l20 * nor.z * nor.z +\n        c4 * sph_arm.l00 -\n        c5 * sph_arm.l20 +\n        2.0 * c1 * sph_arm.l2m2 * nor.x * nor.y +\n        2.0 * c1 * sph_arm.l21  * nor.x * nor.z +\n        2.0 * c1 * sph_arm.l2m1 * nor.y * nor.z +\n        2.0 * c2 * sph_arm.l11  * nor.x +\n        2.0 * c2 * sph_arm.l1m1 * nor.y +\n        2.0 * c2 * sph_arm.l10  * nor.z\n    );\n}\n", 
"atan2", "float atan2(vec2 p) {\n  return atan(p.y, p.x);\n}\n", "noise3D", "float noise3D(vec3 p) {\n  return fract(sin(dot(p ,vec3(12.9898,78.233,12.7235))) * 43758.5453);\n}\n", "worley", "\nfloat create_cells3D(in vec3 p) {\n  p.xy *= 18.0;\n  float d = 1.0e10;\n  vec3 f = floor(p);\n  vec3 x = fract(p);\n  for (int xo = -1; xo <= 1; xo++) {\n    for (int yo = -1; yo <= 1; yo++) {\n      for (int zo = -1; zo <= 1; zo++) {\n        vec3 xyz = vec3(float(xo),float(yo),float(zo));  // Position\n        vec3 tp = xyz + noise3D((xyz+f)/18.0) - x;\n        d = min(d, dot(tp, tp));\n      }\n    }\n  }\n  return sin(d);\n}\n", 
"hash_perlin", "vec2 hash_perlin( vec2 p ) {\n  p = vec2( dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3)) );\n  return -1.0 + 2.0*fract(sin(p)*43758.5453123);\n}\n ", "perlin", " // Based on https://www.shadertoy.com/view/Msf3WH\n  float perlin( in vec2 p ) {\n    const float K1 = 0.366025404; // (sqrt(3)-1)/2;\n    const float K2 = 0.211324865; // (3-sqrt(3))/6;\n    vec2 i = floor( p + (p.x+p.y)*K1 );\n    vec2 a = p - i + (i.x+i.y)*K2;\n    vec2 o = (a.x>a.y) ? vec2(1.0,0.0) : vec2(0.0,1.0); //vec2 of = 0.5 + 0.5*vec2(sign(a.x-a.y), sign(a.y-a.x));\n    vec2 b = a - o + K2;\n    vec2 c = a - 1.0 + 2.0*K2;\n    vec3 h = max( 0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );\n    vec3 n = h*h*h*h*vec3( dot(a,hash_perlin(i+0.0)), dot(b,hash_perlin(i+o)), dot(c,hash_perlin(i+1.0)));\n    return dot( n, vec3(70.0) );\n}\n", 
"sellmeier", "float sellmeierDispersion( float bx, float by, float bz, float cx, float cy, float cz, float wavelength ) {\n      float lams = wavelength * wavelength / 1000000.0;\n      return sqrt( 1.0 + ( bx * lams ) / ( lams - cx ) + ( by * lams ) / ( lams - cy ) + ( bz * lams ) / ( lams - cz ) );\n    }\n    ", "noise", "\n    float rand_(vec2 n) { \n      return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);\n    }\n    float noise(vec2 n) {\n        const vec2 d = vec2(0.0, 1.0);\n      vec2 b = floor(n), f = smoothstep(vec2(0.0), vec2(1.0), fract(n));\n        return mix(mix(rand_(b), rand_(b + d.yx), f.x), mix(rand_(b + d.xy), rand_(b + d.yy), f.x), f.y);\n    }\n    ", 
"read_file", "read_script", "read_text", "uniformLocations", "attribLocations", "shaders", "addAttributes", "mCompiledShader", "getAttribLocation", "addUniforms", "getUniformLocation", "program", "addShader", "loadAndCompileWithFile", "loadAndCompile", "loadAndCompileFromText", "push", "compile_and_link", "createProgram", "attachShader", "linkProgram", "getProgramParameter", "Error in program linking:", "getProgramInfoLog", "warn", "SHADER ERROR", "GET", "open", "send", "ERROR: ", "responseText", 
"WARNING: ", " failed", "compileShader", "textContent", "firstChild", "VERTEX_SHADER", "vertexSource", "FRAGMENT_SHADER", "createShader", "shaderSource", "getShaderParameter", "getShaderInfoLog", "use", "useProgram", "dispose", "FAILED TO LOAD MESH ", "parse", "createTexture", "onload", "bindTexture", "texImage2D", "texParameteri", "onerror", "processId", "forEach", "requiredMaterials", "uniform mat3 rotationMatrix;\n            uniform vec3 cameraPosition;\n          uniform vec2 times;\n           ", 
"getUniformsDeclaration", "getUniforms", "\nvec4 calculateColor( vec2 p ) {\n  vec3 rayPos = cameraPosition;\n  rayPos = rayPos + rotationMatrix * vec3( 0, 0, 1.0 ) * (- 4.0 + 8.0 * ( 2.0 * floor( p.y ) ) );\n", "  vec2 jittered = ( p + ( 1.0 + pow( pseudorandom(seed * 134.16 + 12.6), 6.0 ) * 20.0 ) * ( vec2( pseudorandom(seed * 34.16 + 2.6), pseudorandom(seed * 117.13 + 0.26) ) - 0.5 ) * 1.0 / size ) - 0.5;\n", "  vec2 jittered = ( p + ( vec2( pseudorandom(seed * 34.16 + 2.6), pseudorandom(seed * 117.13 + 0.26) ) - 0.5 ) * 1.0 / size ) - 0.5;\n", 
"computeRayDir", "  vec3 attenuation = vec3( 1.0);\n  vec3 accumulation = vec3( 0.0 );\n  float ior = ", ";\n  float iorNext;\n  vec3 normal;\n  vec3 hitPos;\n  bool inside = false;\n  int bounceType;\n", "getMaterialShader", "  for( int bounce = 0; bounce < ", "; bounce++ ) {\n                   int hitObject = 0;\n                    float t = INFINITY;\n                   inside = false;\n           ", "getIntersectionExprType", " ", "prefix", "hit = ", "rayPos", "rayDir", "getIntersectionExpr", 
";\n", "    if ( ", "hit", "getValidIntersectionCheck", " && ", "getT", " < t ) {\n      t = ", ";\n      hitObject = ", ";\n    }\n", "   hitPos = rayT( rayPos, rayDir, t );\n                 if ( t == INFINITY ) {\n                        bounceType = 0;\n           ", "    } else if ( hitObject == ", " ) {", "getInsideExpression", "false", "      inside = ", "      normal = ", "hitPos", "getNormal", "normal", "getHitStatements", "    }\n    if ( bounceType == 0 ) {\n", "getEnvironmentExpression", 
"      break;\n", "    } else if ( bounceType == ", " ) {\n", "getProcessStatements", "    }\n             }\n             vec3 col = calcIrradiance(normal);\n                accumulation = mix(col*accumulation/", ", col, 0.0);\n                return vec4(accumulation, 1.0);\n           }\n         ", "requiredSourceFrag", "rotationMatrix", "cameraPosition", "times", "uniforms", "uniform mat3 rotationMatrix;\n          uniform vec3 cameraPosition;\n          uniform vec2 times;\n           ", "\n         ", 
"\n           ", "\n       ", "float shadow( vec3 rayPos, vec3 rayDir ) {\n", "    \n                           if ( ", " < 1.0 ) {\n                             return 0.0;\n                           }", "return 1.0;\n}", "\n            vec4 calculateColor( vec2 p ) {\n               vec3 rayPos = cameraPosition;\n             rayPos = rayPos + rotationMatrix * vec3( 0, 0, 1.0 ) * (- 4.0 + 8.0 * ( 2.0 * floor( p.y ) ) );\n           ", "\n             ", "\n             vec3 attenuation = vec3( 1.0 );\n               vec3 accumulation = vec3( 0.0 );\n              float ior = ", 
";\n              float iorNext;\n                vec3 normal;\n              vec3 hitPos;\n              bool inside = false;\n              int bounceType;", "  \n              int hitObject = 0;\n                float t = INFINITY;\n               for( int bounce = 0; bounce < ", "; bounce++ ) {\n                 hitObject = 0;\n                    t = INFINITY;\n                 inside = false;\n           ", ";", "\n                        if ( ", " < t ) {\n                            t = ", 
";\n                           hitObject = ", ";\n                     }", "   hitPos = rayT( rayPos, rayDir, t );\n                   if ( t == INFINITY ) {\n                        bounceType = 0;\n           ", ";\n                                     ", "\n                 }\n                 // hit nothing, environment light\n                 if ( bounceType == 0 ) {\n          ", "\n                      break;", " ) {\n                 ", "     \n                   }\n                 vec3 toLight = sphere1center - hitPos;     \n                   float diffuse = max(0.0, dot(normalize(toLight), normal));    \n                    float shadowIntensity = shadow(hitPos + normal * 0.0001, toLight);\n                    accumulation += attenuation * (diffuse * shadowIntensity);\n                }\n             return vec4( accumulation / ", 
", 1 );\n          }\n         ", "loadMesh", "loadTexture", "createSphericalHarmonicShader", "createSceneProgram", "time", "weight", "previousTexture", "size", "#version 300 es\n      precision highp float;\n\n      precision highp int;\n      precision highp isampler2D;\n\n     in vec2 texCoord;\n     uniform float time;\n       uniform float weight;\n     uniform vec2 size;\n        uniform sampler2D previousTexture;\n\n      out vec4 fragColor;\n       #define INFINITY 10000.0\n\n        float seed;\n\n     ", 
"\n     \n      void main( void ) {\n           vec4 previous = vec4( texture( previousTexture, gl_FragCoord.xy / size ).rgb, 1 );\n            vec4 sample_ = vec4(0);\n           for( int i = 0; i < ", "; i++) {\n             seed = time + float(i);\n               sample_ = sample_ + calculateColor( texCoord );\n           }\n         fragColor = mix( sample_ / ", ", previous, weight );\n        }", "vertex", "depthField", "uniform float focalLength;\n      uniform float depthField;\n      ", "\n        vec2 dofOffset = depthField * uniformInsideDisk( pseudorandom(seed * 92.72 + 2.9), pseudorandom(seed * 192.72 + 12.9) );\n        vec3 rayDir = rotationMatrix * normalize( vec3( jittered - dofOffset / focalLength, 1.0 ) );\n        rayPos += rotationMatrix * vec3( dofOffset, 0.0 );\n      ", 
"uniform1f", "getRayDir", "fromValues", "PerspectiveRays", "B", "origin", "direction", "point_at_parameter", "Ray", "__extends", "constructor", "hasOwnProperty", "create", "AbsEnviroment", "call", "SimpleEnvironment", "envTexture", "envRotation", "envMult", "activeTexture", "uniform1i", "\n        uniform sampler2D envTexture;\n        uniform float envRotation;\n        uniform float envMult;", "vec2( -atan( rayDir.z, rayDir.x ) / TWO_PI + 0.5 + envRotation, ( 0.5 - asin( rayDir.y ) / PI ) * 2.0)", 
"texture( envTexture, ", " ).rgb", "      accumulation += attenuation * ", " * envMult;\n", "TextureEnvironment", "movSpeed", "mouseSensivity", "view", "proj", "front", "position", "worldUp", "yaw", "pitch", "right", "up", "updateCameraVectors", "GetPos", "scaleAndAdd", "toRadian", "cos", "sin", "normalize", "cross", "GetViewMatrix", "add", "lookAt", "GetProjectionMatrix", "perspective", "renderableGlobalID", "isDynamic", "box", "isTwoSided", "min", "max", "minName", "maxName", "centerName", "halfSizeName", 
"halfSize", "uniform3fv", "uniform vec3 ", ";\nuniform vec3 ", ";\nvec3 ", " = ( ", " + ", " ) / 2.0;\nvec3 ", " - ", " ) / 2.0;\n", "vec2", "rayIntersectBox( ", ", ", " );", "(", ".y > ", ".x < ", ".y)", ".x > ", " ? ", ".x : ", ".x", ".x < 0.0)", "sub", "( sign( ", ".x ) * normalForBox( ", " ) )", "normalForBox( ", " )", "hitRay", "POSITIVE_INFINITY", "sphere", "radius", "radiusName", "radiusValue", "centerValue", "( ", " + vec3( -( times.x + ( times.y - times.x ) * ( pseudorandom(seed*14.53+1.6) ) ), 0.0, 0.0 ) )", 
";\nuniform float ", "rayIntersectSphere( ", ".x ) * normalForSphere( ", "normalForSphere( ", "subtract", "dot", "sqrt", "color", "r", "g", "b", "plane", "d", "normalName", "dName", "const vec3 ", " = ", ";\nconst float ", "float", "rayIntersectPlane( ", " > ", "triangle", "v0", "v1", "v2", "intersectTriangle(", ");", "normalForTriangle(", "mesh", "vertices", "indices", "aabb", "vertex_positions", "triangles_list", "VERTEX_TEX_SIZE", "TRIANGLE_TEX_SIZE", "meshVerticesBuffer", "createBuffer", "bindBuffer", 
"bufferData", "meshIndicesBuffer", "texVertices", "texTriangles", "\n          uniform sampler2D vertex_positions;\n           uniform isampler2D triangles_list;\n            #define VERTEX_TEX_SIZE 17    //vertices\n          #define TRIANGLE_TEX_SIZE 6  //indices\n\n          const ivec3 indices[", "] = ivec3[](", "ivec3(", "),", "substring", ");\n            vec2 intersectMesh1(vec3 origin, vec3 dir) {\n\n                float t = INFINITY;\n               int tri = -3;\n             ivec3 list_pos;\n               vec3 v0, v1, v2;\n              float hit;\n                int index = 0;\n\n              vec2 aabb_hit = rayIntersectBox(", 
"cubeMin", ", \n                        ", "cubeMax", ", origin, dir);\n               if ((aabb_hit.x > ", " && aabb_hit.x < aabb_hit.y) && aabb_hit.x < t) {\n                   ", "list_pos = indices[index++]; //texture(triangles_list, vec2((float(index++) + 0.5) / float(", "), 0.5)).xyz;\n                   if ((index + 1) % 2 != 0) {\n                       list_pos.xyz = list_pos.zxy;\n                  }\n                 v0 = (texture(vertex_positions, vec2((float(list_pos.z) + 0.5) / float(", 
"), 0.5)).xyz);\n                    v1 = (texture(vertex_positions, vec2((float(list_pos.y) + 0.5) / float(", "), 0.5)).xyz);\n                    v2 = (texture(vertex_positions, vec2((float(list_pos.x) + 0.5) / float(", "), 0.5)).xyz);\n\n                  hit = intersectTriangle(origin, dir, v0, v1, v2);\n                 if (hit < t) {\n                        t = hit;\n                      tri = index;\n                  }", "\n               }\n             return vec2(t, float(tri / 3));\n           }", 
"intersectObjMesh", "MAX_VALUE", "MIN_VALUE", "uniform1ui", "intersectMesh1(", "normalOnMesh( ", ".y);", "mSceneXML", "text/xml", "parseFromString", "SceneLevel", "getElementsByTagName", "parseCamera", "parseSceneLight", "Objects", "parseSpheres", "parseBoxes", "parseTriangles", "parseMeshes", "parsePlanes", "parseLights", "str2FloatArr", "createMaterial", "diffuse", "specular", "Textured", "Absorb", "tagName", "firstElementChild", "intensity", "string2Float", "Emit", "Diffuse", "Attenuate", "childElementCount", 
"childNodes", "na", "nb", "FresnelComposite", "glossiness", "montecarlo", "iorComplex", "Metal", "getNamedItem", "attributes", "SmoothDielectric", "n", "PhongSpecular", "color1", "color2", "ChessTexture", "PerlinTexture", "FormulaPowTexture", "WorleyTexture", "outerIor", "innerIor", "Transmit", "Reflect", "Camera", "Light", "Sphere", "Box", "Triangle", "Mesh", "Plane", "materialGlobalId", "processGlobalId", "Material", "bounceType = ", "break;", "\n        rayDir = reflect( rayDir, normal );\n        rayPos = hitPos + ", 
" * rayDir;", "\n        transmitIOR", " = vec2( ", " );\n        if ( inside ) {\n          transmitIOR", " = transmitIOR", ".yx;\n        }\n        bounceType = ", "vec2 transmitIOR", "      \n        rayDir = refract( rayDir, normal, transmitIOR", ".x / transmitIOR", ".y );\n        // Total internal reflection case not handled.\n        if ( dot( rayDir, rayDir ) == 0.0 ) { break; }\n        rayPos = hitPos + ", "formula", "\n        rayDir = sampleTowardsNormal( normal, sampleDotWeightOnHemiphere( pseudorandom(float(bounce) + seed*164.32+2.5), pseudorandom(float(bounce) + 7.233 * seed + 1.3) ) );\n        rayPos = hitPos + 0.0001 * rayDir;\n      ", 
"FormulaTextured", "attenuation *= pow( abs( 1.0 + sin( 10.0 * sin( hitPos.x ) ) + sin( 10.0 *sin( hitPos.z ) ) ), 7.0 ) * 0.15;", "\n      // Chess texture based on Shirley books\n      float sines = sin(10.0*hitPos.x * 0.01) * sin(10.0*hitPos.y * 0.01) * sin(10.0*hitPos.z * 0.01);\n      if(sines < 0.0) {\n        attenuation = vec3(", ");\n      } else {\n        attenuation = vec3(", ");\n      }", "ChessTextured", "\n        float phi = atan2(vec2(hitPos.z, hitPos.x));\n        float theta = asin(hitPos.y);\n        vec2 uv;\n        uv.x = 1.0-(phi + PI) / TWO_PI;\n        uv.y = (theta + PI/2.0) / PI;\n\n        float f = perlin(hitPos.xz*0.01);\n        f = 0.5 + 0.5*f;\n\n        attenuation = vec3(f);\n      ", 
"\n        float phi = atan2(vec2(hitPos.z, hitPos.x));\n        float theta = asin(hitPos.y);\n        vec2 uv;\n        uv.x = 1.0-(phi + PI) / TWO_PI;\n        uv.y = (theta + PI/2.0) / PI;\n\n        float f = create_cells3D(vec3(hitPos.xz*0.004, 1.0));\n        attenuation = vec3(1.0)*f*f;\n        attenuation = mix(attenuation, ", ");    // Background\n        attenuation = mix(attenuation, ", ", f);  \n      ", "diffuseTexture", "specularTexture", "normalTexture", "albedoTex", "specularTex", 
"normalTex", "uniform sampler2D albedoTex;\n        uniform sampler2D specularTex;\n        uniform sampler2D normalTex;", "\n        bool inside = abs( hitPos.x ) <= 350.0 && abs( hitPos.z ) <= 350.0;\n        if ( inside ) {\n          vec3 normal2 = normalize( texture( normalTex, hitPos.xz * 0.003937007874015748 ).rbg * 2.0 - 1.0 );\n          vec3 diffuse = pow( abs( texture( albedoTex, hitPos.xz * 0.003937007874015748 ).rgb ), vec3( 2.2 ) );\n          if ( dot( normal2, rayDir ) > 0.0 ) { normal2 = normal; }\n          vec3 transmitDir = refract( rayDir, normal, 1.0 / 1.5 );\n          vec2 reflectance = fresnelDielectric( rayDir, normal, transmitDir, 1.0, 1.5 );\n          bool didReflect = false;\n          if ( pseudorandom(float(bounce) + seed*1.17243 - 2.3 ) < ( reflectance.x + reflectance.y ) / 2.0 ) {\n            vec3 dirtTex = pow( abs( texture( specularTex, hitPos.xz * 0.003937007874015748 ).rgb ), vec3( 2.2 ) ) * 0.4 + 0.6;\n            attenuation *= dirtTex * ( pow( abs( diffuse.y ), 1.0 / 2.2 ) );\n            vec3 reflectDir = reflect( rayDir, normal2 );\n            rayDir = sampleTowardsNormal( reflectDir, sampleDotWeightOnHemiphere( pseudorandom(float(bounce) + seed*1642.32+2.52 - 2.3), pseudorandom(float(bounce) + 72.233 * seed + 1.32 - 2.3) ) );\n            if ( rayDir.y > 0.0 ) {\n              didReflect = true;\n              float reflectDot = dot( reflectDir, rayDir );\n              if ( reflectDot < 0.0 ) { break; }\n              // compute contribution based on normal cosine falloff (not included in sampled direction)\n              float contribution = pow( abs( reflectDot ), 50.0 ) * ( 50.0 + 2.0 ) / ( 2.0 );\n              // if the contribution is negative, we sampled behind the surface (just ignore it, that part of the integral is 0)\n              // weight this sample by its contribution\n              attenuation *= contribution;\n            }\n          }\n          if ( !didReflect ) {\n            attenuation *= diffuse;\n            rayDir = sampleTowardsNormal( normal2, sampleDotWeightOnHemiphere( pseudorandom(float(bounce) + seed*164.32+2.5) - 2.3, pseudorandom(float(bounce) + 7.233 * seed + 1.3 - 2.3) ) );\n            if ( rayDir.y < 0.0 ) { rayDir.y = -rayDir.y; };\n          }\n        } else {\n          attenuation *= pow( vec3( 74.0, 112.0, 25.0 ) / 255.0, vec3( 1.0 / 2.2 ) ) * 0.5;\n          rayDir = sampleTowardsNormal( normal, sampleDotWeightOnHemiphere( pseudorandom(float(bounce) + seed*164.32+2.5) - 2.3, pseudorandom(float(bounce) + 7.233 * seed + 1.3 - 2.3) ) );\n        }\n        rayPos = hitPos + ", 
" * rayDir;\n      ", "iorNext = inside ? ", " : ", ";\n        bounceType = ", "\n        if ( abs( dot( normal, rayDir ) ) < totalInternalReflectionCutoff( ior, iorNext ) + ", " ) {\n          rayDir = reflect( rayDir, normal );\n        } else {\n          vec3 transmitDir = refract( rayDir, normal, ior / iorNext );\n          vec2 reflectance = fresnelDielectric( rayDir, normal, transmitDir, ior, iorNext );\n          if ( pseudorandom(float(bounce) + seed*1.7243 - 15.34) > ( reflectance.x + reflectance.y ) / 2.0 ) {\n            rayDir = transmitDir;      \n            attenuation *= ", 
";\n            ior = iorNext;\n          } else {                     \n            rayDir = reflect( rayDir, normal );\n          }\n        }\n        rayPos = hitPos + ", "nName", "phongN", "uniform float ", "const float ", "\n        phongSpecularN", "float phongSpecularN", "\n              vec3 reflectDir = reflect( rayDir, normal );\n              rayDir = sampleTowardsNormal( reflectDir, sampleDotWeightOnHemiphere( pseudorandom(float(bounce) + seed*1642.32+2.52), pseudorandom(float(bounce) + 72.233 * seed + 1.32) ) );\n              float dotReflectDirWithRay = dot( reflectDir, rayDir );\n              float contribution = pow( abs( dotReflectDirWithRay ), phongSpecularN", 
" ) * ( phongSpecularN", " + 2.0 ) / ( 2.0 );\n              if ( dotReflectDirWithRay < 0.0 ) { break; }\n              attenuation *= contribution;\n              rayPos = hitPos + ", "sampleMethod", " iorComplex", ";\n        glossiness", "\n        vec2 iorComplex", ";\n        float glossiness", ";\n      ", " vec2 reflectance = fresnel( rayDir, normal, ior, iorComplex", ".x, iorComplex", ".y );\n      attenuation *= ", " * ( reflectance.x + reflectance.y ) / 2.0;\n      rayDir = reflect( rayDir, normal ) + glossiness", 
" * ", "( pseudorandom(float(bounce) + seed*164.32+2.5), pseudorandom(float(bounce) + 7.233 * seed + 1.3) );\n      // sampleUniformOnHemisphere: Uniform pseudorandom samples.\n      // sampleDotWeightOnHemiphere: Dot-weighted pseudorandom samples calculated using a Monte Carlo method.\n      rayPos = hitPos + ", "materialA", "materialB", "float ratio", "ratioStatements", "if ( pseudorandom(float(bounce) + seed*1.7243 - float(", ") ) < ratio", "} else {\n", "}\n", "SwitchedMaterial", "\n        vec2 fresnelIors = vec2( ", 
");\n        if ( inside ) { fresnelIors = fresnelIors.yx; }\n        if ( abs( dot( normal, rayDir ) ) < totalInternalReflectionCutoff( fresnelIors.x, fresnelIors.y ) + ", " ) {\n          ratio", " = 1.0;\n        } else {;\n          vec2 reflectance = fresnelDielectric( rayDir, normal, refract( rayDir, normal, fresnelIors.x / fresnelIors.y ), fresnelIors.x, fresnelIors.y );\n          ratio", " = ( reflectance.x + reflectance.y ) / 2.0;\n        }", "statements", "WrapperMaterial", "attenuation *= ", 
"attenuationName", "attenuation", "x", "y", "z", "uniform3f", "accumulation += attenuation * ", "emissionName", "color;", "\n          uniform vec3 ", ";\n          uniform vec3 ", "\n          const vec3 ", ";\n          const vec3 ", "color = ", "\n        rayDir = sampleTowardsNormal( normal, ", "( pseudorandom(float(bounce) + seed*164.32+2.5), pseudorandom(float(bounce) + 7.233 * seed + 1.3) ) );\n        rayPos = hitPos + ", "mKeyPreviousState", "mIsKeyPressed", "mIsKeyClicked", "_instance", 
"initialize", "LastKeyCode", "keyup", "onmousedown", "onmouseup", "_onKeyUp", "_onKeyDown", "_onMouseDown", "_onMouseUp", "isKeyPressed", "isKeyClicked", "pixelStorei", "responseType", "arraybuffer", "response", "pow", "createHDRTexture", "#version 300 es\nprecision highp float;\nin vec2 texCoord;\nuniform sampler2D texture_;\nout vec4 fragColor;\nvoid main() {\n  fragColor = texture( texture_, texCoord );\n}", "texture_", "#version 300 es\nprecision highp float;\nin vec2 texCoord;\nuniform sampler2D texture_;\nuniform float brightness;\nout vec4 fragColor;\nvoid main() {\n  fragColor = texture( texture_, texCoord );\n  fragColor.rgb = brightness * pow( abs( fragColor.rgb ), vec3( 1.0 / 2.2 ) );\n}", 
"#version 300 es\nprecision highp float;\nin vec2 texCoord;\nuniform sampler2D texture_;\nuniform float brightness;\nout vec4 fragColor;\nvoid main() {\n  fragColor = texture( texture_, texCoord );\n  fragColor.rgb = fragColor.rgb / ( 1.0 + fragColor.rgb );\n  fragColor.rgb = brightness * pow( abs( fragColor.rgb ), vec3( 1.0 / 2.2 ) );\n}", "#version 300 es\nprecision highp float;\nin vec2 texCoord;\nuniform sampler2D texture_;\nuniform float brightness;\nout vec4 fragColor;\nvoid main() {\n  vec3 color = texture( texture_, texCoord ).rgb * pow( abs( brightness ), 2.2 );\n  color = max(vec3(0.), color - vec3(0.004));\n  color = (color * (6.2 * color + .5)) / (color * (6.2 * color + 1.7) + 0.06);\n  fragColor = vec4( color, 1.0 );\n}", 
"#version 300 es\nprecision highp float;\nin vec2 texCoord;\nuniform sampler2D texture_;\nout vec4 fragColor;\nfloat sRGB_gamma_correct(float c) {\n const float a = 0.055;\n if(c < 0.0031308) return 12.92*c;\n else return (1.0+a)*pow(c, 1.0/2.4) - a;\n}\nvoid main() {\n  fragColor = texture( texture_, texCoord );\n  fragColor.r = sRGB_gamma_correct(fragColor.r);\n  fragColor.g = sRGB_gamma_correct(fragColor.g);\n  fragColor.b = sRGB_gamma_correct(fragColor.b);\n}", "#version 300 es\nprecision highp float;\nin vec2 texCoord;\nuniform sampler2D texture_;\nuniform float brightness;\nout vec4 fragColor;\nvoid main() {\n  fragColor = texture( texture_, texCoord );\n  float A = 0.15;\n  float B = 0.50;\n  float C = 0.10;\n  float D = 0.20;\n  float E = 0.02;\n  float F = 0.30;\n  float W = 11.2;\n  float exposure = brightness;//2.;\n  fragColor.rgb *= exposure;\n  fragColor.rgb = ((fragColor.rgb * (A * fragColor.rgb + C * B) + D * E) / (fragColor.rgb * (A * fragColor.rgb + B) + D * F)) - E / F;\n  float white = ((W * (A * W + C * B) + D * E) / (W * (A * W + B) + D * F)) - E / F;\n  fragColor.rgb /= white;\n  fragColor.rgb = pow(fragColor.rgb, vec3(1. / 2.2));\n}", 
"uniformCallback", "renderProgram", "startTime", "recreateShader", "makeTexture", "FLOAT", "vertexBuffer", "frameBuffer", "createFramebuffer", "textures", "samples", "deleteBuffer", "deleteFramebuffer", "state0", "state1", "uniform2fv", "bindFramebuffer", "framebufferTexture2D", "enableVertexAttribArray", "vertexAttribPointer", "drawArrays", "reverse", "SourceFrag not found", "scene0", "getTime", "ceil", "reduce", "round", " FPS", "fps", "uniformMatrix3fv", "uniform2f", "load", "Error: "];
/**
 * @return {?}
 */
function getContext() {
  var context;
  var name;
  var configList = _0x398e[2][_0x398e[1]](_0x398e[0]);
  /** @type {number} */
  var i = 0;
  for (;i < configList[_0x398e[3]];i++) {
    if (name = configList[i], gl = canvas[_0x398e[4]](name)) {
      context = name;
      break;
    }
  }
  return context;
}
/**
 * @return {undefined}
 */
function getVendors() {
  var cn = _0x398e[5][_0x398e[1]](_0x398e[0]);
  if (!window[_0x398e[6]]) {
    var n;
    /** @type {number} */
    var x = 0;
    for (;x < cn[_0x398e[3]] && (n = cn[x], window[_0x398e[6]] = window[n + _0x398e[7]], window[_0x398e[8]] = window[n + _0x398e[9]] || window[n + _0x398e[10]], !window[_0x398e[6]]);x++) {
    }
  }
}
/**
 * @param {?} dataAndEvents
 * @return {undefined}
 */
function updateXML(dataAndEvents) {
  /**
   * @param {(number|string)} obj
   * @return {?}
   */
  function isArray(obj) {
    var _0x6f51x2 = obj.toString(16);
    return 1 == _0x6f51x2[_0x398e[3]] ? _0x398e[11] + _0x6f51x2 : _0x6f51x2;
  }
  /**
   * @param {Array} dataAndEvents
   * @return {?}
   */
  function clone(dataAndEvents) {
    return _0x398e[12] + isArray(255 * dataAndEvents[0]) + isArray(255 * dataAndEvents[1]) + isArray(255 * dataAndEvents[2]);
  }
  Renderable[_0x398e[13]]();
  if (sl) {
    delete sl[_0x398e[14]];
    delete sl[_0x398e[15]];
    delete sl[_0x398e[16]];
    delete sl[_0x398e[17]];
    delete sl[_0x398e[18]];
  }
  /** @type {boolean} */
  loading = true;
  sl = new SceneLoader(_0x398e[19] + dataAndEvents + _0x398e[20]);
  lightSphere = sl[_0x398e[21]];
  document[_0x398e[24]](_0x398e[23])[_0x398e[22]] = lightSphere[_0x398e[25]][0];
  document[_0x398e[24]](_0x398e[26])[_0x398e[22]] = lightSphere[_0x398e[25]][1];
  document[_0x398e[24]](_0x398e[27])[_0x398e[22]] = lightSphere[_0x398e[25]][2];
  camera = sl[_0x398e[28]];
  document[_0x398e[24]](_0x398e[29])[_0x398e[22]] = clone(lightSphere[_0x398e[30]]()) + _0x398e[31];
  console[_0x398e[32]](clone(lightSphere[_0x398e[30]]()));
  /** @type {Array} */
  currentSceneObjects = [lightSphere];
  currentSceneObjects = currentSceneObjects[_0x398e[33]](sl[_0x398e[14]]);
  currentSceneObjects = currentSceneObjects[_0x398e[33]](sl[_0x398e[15]]);
  currentSceneObjects = currentSceneObjects[_0x398e[33]](sl[_0x398e[18]]);
  currentSceneObjects = currentSceneObjects[_0x398e[33]](sl[_0x398e[16]]);
  currentSceneObjects = currentSceneObjects[_0x398e[33]](sl[_0x398e[17]]);
  currentSceneObjects = currentSceneObjects[_0x398e[33]](sl[_0x398e[34]]);
}
/**
 * @return {undefined}
 */
function initializeTracer() {
  tracerIntegrator = new PathTracer(integratorCallBack, size);
}
/**
 * @return {undefined}
 */
function updateStepProgram() {
  /** @type {boolean} */
  loading = true;
  setStatus(_0x398e[35]);
  if (!tracerIntegrator) {
    initializeTracer();
  }
  try {
    console[_0x398e[32]](_0x398e[36]);
    tracerIntegrator[_0x398e[37]](currentSceneObjects, projection, environment, bounces);
    console[_0x398e[32]](_0x398e[38]);
    /** @type {boolean} */
    loading = false;
  } catch (udataCur) {
    throw setStatus(udataCur), console[_0x398e[32]](tracerIntegrator[_0x398e[40]][_0x398e[39]]), udataCur;
  }
  tracerIntegrator[_0x398e[41]]();
}
/**
 * @param {?} deepDataAndEvents
 * @param {number} replacementHash
 * @param {number} expectedNumberOfNonCommentArgs
 * @param {number} opt_attributes
 * @return {undefined}
 */
function loadEnvironmentTexture(deepDataAndEvents, replacementHash, expectedNumberOfNonCommentArgs, opt_attributes) {
  /** @type {boolean} */
  loading = true;
  gl[_0x398e[42]](envTexture);
  setStatus(_0x398e[43]);
  HDRLoader[_0x398e[44]](gl, deepDataAndEvents, replacementHash, expectedNumberOfNonCommentArgs, gl.RGB, function(dataAndEvents) {
    envTexture = dataAndEvents;
    setStatus();
    environment = new environments.TextureEnvironment(envTexture, opt_attributes, environmentRotation);
    updateStepProgram();
  });
}
/**
 * @return {undefined}
 */
function initUI() {
  var _0x6f51x4 = document[_0x398e[24]](_0x398e[45]);
  _0x6f51x4[_0x398e[47]](_0x398e[46], function() {
    /** @type {boolean} */
    loading = true;
    /** @type {boolean} */
    fog = !fog;
    console[_0x398e[32]](fog);
    updateStepProgram();
  });
  var _0x6f51x2 = document[_0x398e[24]](_0x398e[48]);
  var test = document[_0x398e[24]](_0x398e[49]);
  test[_0x398e[47]](_0x398e[50], function() {
    /** @type {number} */
    tracerIntegrator[_0x398e[51]] = parseFloat(test[_0x398e[22]]) / 100;
  });
  var t = document[_0x398e[24]](_0x398e[52]);
  t[_0x398e[47]](_0x398e[50], function() {
    /** @type {number} */
    projection[_0x398e[53]] = parseFloat(t[_0x398e[22]]) / 25;
    tracerIntegrator[_0x398e[41]]();
  });
  var coords = document[_0x398e[24]](_0x398e[54]);
  coords[_0x398e[47]](_0x398e[50], function() {
    /** @type {number} */
    projection[_0x398e[55]] = parseFloat(coords[_0x398e[22]]);
    tracerIntegrator[_0x398e[41]]();
  });
  var explodedMatrix = document[_0x398e[24]](_0x398e[23]);
  explodedMatrix[_0x398e[47]](_0x398e[50], function() {
    /** @type {number} */
    lightSphere[_0x398e[25]][0] = parseFloat(explodedMatrix[_0x398e[22]]);
    lightSphere[_0x398e[56]]();
    tracerIntegrator[_0x398e[41]]();
  });
  var boo = document[_0x398e[24]](_0x398e[26]);
  boo[_0x398e[47]](_0x398e[50], function() {
    /** @type {number} */
    lightSphere[_0x398e[25]][1] = parseFloat(boo[_0x398e[22]]);
    lightSphere[_0x398e[56]]();
    tracerIntegrator[_0x398e[41]]();
  });
  var offset = document[_0x398e[24]](_0x398e[27]);
  offset[_0x398e[47]](_0x398e[50], function() {
    /** @type {number} */
    lightSphere[_0x398e[25]][2] = parseFloat(offset[_0x398e[22]]);
    lightSphere[_0x398e[56]]();
    tracerIntegrator[_0x398e[41]]();
  });
  var $cookies = document[_0x398e[24]](_0x398e[29]);
  $cookies[_0x398e[47]](_0x398e[50], function() {
    /**
     * @param {Text} element
     * @return {?}
     */
    function post(element) {
      /** @type {RegExp} */
      var type = /^#?([a-f\d])([a-f\d])([a-f\d])$/i;
      element = element[_0x398e[57]](type, function(dataAndEvents, $1, $2, times) {
        return $1 + $1 + $2 + $2 + times + times;
      });
      var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i[_0x398e[58]](element);
      return result ? {
        r : parseInt(result[1], 16) / 255,
        g : parseInt(result[2], 16) / 255,
        b : parseInt(result[3], 16) / 255
      } : null;
    }
    var value = $cookies[_0x398e[22]];
    lightSphere[_0x398e[59]](post(value));
    tracerIntegrator[_0x398e[41]]();
  });
  var size = document[_0x398e[24]](_0x398e[60]);
  size[_0x398e[22]] = lightSphere[_0x398e[62]][_0x398e[61]][0];
  size[_0x398e[47]](_0x398e[50], function() {
    lightSphere[_0x398e[63]](parseFloat(size[_0x398e[22]]));
    tracerIntegrator[_0x398e[41]]();
  });
  var values = document[_0x398e[24]](_0x398e[64]);
  values[_0x398e[22]] = globals[_0x398e[65]] + _0x398e[31];
  var _0x6f51x13 = document[_0x398e[24]](_0x398e[66]);
  _0x6f51x13[_0x398e[47]](_0x398e[46], function() {
    /** @type {number} */
    globals[_0x398e[65]] = parseFloat(values[_0x398e[22]]);
    setStatus(_0x398e[67]);
    updateStepProgram();
  });
  var _0x6f51x14 = document[_0x398e[24]](_0x398e[68]);
  _0x6f51x14[_0x398e[47]](_0x398e[46], function() {
    var text = tracerIntegrator[_0x398e[40]][_0x398e[39]];
    /** @type {Blob} */
    var source = new Blob([text], {
      type : _0x398e[69]
    });
    var key = window[_0x398e[71]][_0x398e[70]](source);
    var res = document[_0x398e[73]](_0x398e[72]);
    res[_0x398e[74]] = _0x398e[75];
    res[_0x398e[76]] = key;
    res[_0x398e[46]]();
  });
  var _0x6f51x15 = document[_0x398e[24]](_0x398e[37]);
  _0x6f51x15[_0x398e[47]](_0x398e[46], function() {
    console[_0x398e[32]](_0x398e[77]);
    initializeTracer();
    updateStepProgram();
  });
  var strings = document[_0x398e[24]](_0x398e[78]);
  strings[_0x398e[47]](_0x398e[50], function() {
    /** @type {number} */
    bounces = parseInt(strings[_0x398e[22]]);
    setStatus(_0x398e[79]);
    updateStepProgram();
  });
  var cols = document[_0x398e[24]](_0x398e[80]);
  cols[_0x398e[47]](_0x398e[50], function() {
    /** @type {number} */
    tracerIntegrator[_0x398e[81]] = parseInt(cols[_0x398e[22]]);
    tracerIntegrator[_0x398e[41]]();
  });
  var _0x6f51x18 = document[_0x398e[24]](_0x398e[82]);
  _0x6f51x18[_0x398e[47]](_0x398e[46], function() {
    _0x6f51x2[_0x398e[84]][_0x398e[83]] = _0x398e[85];
    tracerIntegrator[_0x398e[87]](toneMap[_0x398e[86]]);
  });
  var _0x6f51x19 = document[_0x398e[24]](_0x398e[88]);
  _0x6f51x19[_0x398e[47]](_0x398e[46], function() {
    _0x6f51x2[_0x398e[84]][_0x398e[83]] = _0x398e[89];
    tracerIntegrator[_0x398e[87]](toneMap[_0x398e[90]]);
  });
  var _0x6f51x1a = document[_0x398e[24]](_0x398e[91]);
  _0x6f51x1a[_0x398e[47]](_0x398e[46], function() {
    _0x6f51x2[_0x398e[84]][_0x398e[83]] = _0x398e[89];
    tracerIntegrator[_0x398e[87]](toneMap[_0x398e[92]]);
  });
  var _0x6f51x1b = document[_0x398e[24]](_0x398e[93]);
  _0x6f51x1b[_0x398e[47]](_0x398e[46], function() {
    _0x6f51x2[_0x398e[84]][_0x398e[83]] = _0x398e[89];
    tracerIntegrator[_0x398e[87]](toneMap[_0x398e[94]]);
  });
  var _0x6f51x1c = document[_0x398e[24]](_0x398e[95]);
  _0x6f51x1c[_0x398e[47]](_0x398e[46], function() {
    _0x6f51x2[_0x398e[84]][_0x398e[83]] = _0x398e[85];
    tracerIntegrator[_0x398e[87]](toneMap[_0x398e[96]]);
  });
  var v = document[_0x398e[24]](_0x398e[97]);
  v[_0x398e[47]](_0x398e[46], function() {
    _0x6f51x2[_0x398e[84]][_0x398e[83]] = _0x398e[89];
    tracerIntegrator[_0x398e[87]](toneMap[_0x398e[98]]);
  });
  var _0x6f51x1e = document[_0x398e[24]](_0x398e[99]);
  _0x6f51x1e[_0x398e[47]](_0x398e[46], function() {
    environment = new environments.SimpleEnvironment(_0x398e[100], []);
    updateStepProgram();
  });
  var s = document[_0x398e[24]](_0x398e[101]);
  s[_0x398e[47]](_0x398e[50], function() {
    /** @type {number} */
    environment[_0x398e[102]] = environmentRotation = parseFloat(s[_0x398e[22]]) / 100;
    tracerIntegrator[_0x398e[41]]();
  });
  var components = document[_0x398e[24]](_0x398e[103]);
  components[_0x398e[47]](_0x398e[50], function() {
    /** @type {number} */
    environment[_0x398e[104]] = parseFloat(components[_0x398e[22]]);
    tracerIntegrator[_0x398e[41]]();
  });
  var matches = document[_0x398e[24]](_0x398e[105]);
  matches[_0x398e[47]](_0x398e[50], function() {
    /** @type {number} */
    startTime = parseFloat(matches[_0x398e[22]]);
    tracerIntegrator[_0x398e[41]]();
  });
  var matrix = document[_0x398e[24]](_0x398e[106]);
  matrix[_0x398e[47]](_0x398e[50], function() {
    /** @type {number} */
    shutterOpenTime = parseFloat(matrix[_0x398e[22]]);
    tracerIntegrator[_0x398e[41]]();
  });
  document[_0x398e[24]](_0x398e[109])[_0x398e[47]](_0x398e[46], function() {
    loadEnvironmentTexture(_0x398e[107], 2048, 512, 2);
    components[_0x398e[22]] = _0x398e[108];
  });
  document[_0x398e[24]](_0x398e[112])[_0x398e[47]](_0x398e[46], function() {
    loadEnvironmentTexture(_0x398e[110], 2048, 512, 0.5);
    components[_0x398e[22]] = _0x398e[111];
  });
  document[_0x398e[24]](_0x398e[115])[_0x398e[47]](_0x398e[46], function() {
    loadEnvironmentTexture(_0x398e[113], 2048, 512, 1.2);
    components[_0x398e[22]] = _0x398e[114];
  });
  var _0x6f51x23 = document[_0x398e[24]](_0x398e[116]);
  _0x6f51x23[_0x398e[47]](_0x398e[117], function(dataAndEvents) {
    setStatus(_0x398e[118]);
    updateXML(_0x6f51x23[_0x398e[22]]);
    updateStepProgram();
  });
  var octalLiteral = document[_0x398e[24]](_0x398e[119]);
  octalLiteral[_0x398e[47]](_0x398e[117], function(dataAndEvents) {
    switch(setStatus(_0x398e[129]), parseInt(octalLiteral[_0x398e[22]])) {
      case 0:
        console[_0x398e[32]](_0x398e[120]);
        console[_0x398e[32]](_0x398e[121]);
        console[_0x398e[32]](_0x398e[122]);
        break;
      case 1:
        console[_0x398e[32]](_0x398e[123]);
        tracerIntegrator = new PathTracer(integratorCallBack, size);
        break;
      case 2:
        console[_0x398e[32]](_0x398e[124]);
        console[_0x398e[32]](octalLiteral[_0x398e[128]][_0x398e[127]](0)[_0x398e[126]](_0x398e[125]));
        tracerIntegrator = new SphericalHarmonics(integratorCallBack, size, octalLiteral[_0x398e[128]][_0x398e[127]](0)[_0x398e[126]](_0x398e[125]));
    }
    updateStepProgram();
  });
}
/**
 * @param {?} value
 * @return {undefined}
 */
function setStatus(value) {
  var flags = document[_0x398e[24]](_0x398e[130]);
  flags[_0x398e[131]] = value;
  flags[_0x398e[132]][_0x398e[84]][_0x398e[83]] = value ? _0x398e[89] : _0x398e[85];
}
/**
 * @return {undefined}
 */
function initialize() {
  /** @type {number} */
  startTime = 0;
  /** @type {number} */
  shutterOpenTime = 0;
  /** @type {number} */
  environmentRotation = parseFloat(document[_0x398e[24]](_0x398e[101])[_0x398e[22]]) / 360;
  environment = new environments.SimpleEnvironment(_0x398e[133], []);
  projection = new scene.PerspectiveRays(50, 0);
  document[_0x398e[47]](_0x398e[134], function(dataAndEvents) {
    if (40 === dataAndEvents[_0x398e[135]] || 38 === dataAndEvents[_0x398e[135]]) {
      dataAndEvents[_0x398e[136]]();
    }
    var _0x6f51x2 = String[_0x398e[137]](dataAndEvents[_0x398e[135]]);
    /** @type {number} */
    var r20 = 0.5;
    switch(_0x6f51x2) {
      case _0x398e[139]:
        camera[_0x398e[138]](4, r20);
        tracerIntegrator[_0x398e[41]]();
        break;
      case _0x398e[140]:
        camera[_0x398e[138]](5, r20);
        tracerIntegrator[_0x398e[41]]();
        break;
      case _0x398e[141]:
        camera[_0x398e[138]](2, r20);
        tracerIntegrator[_0x398e[41]]();
        break;
      case _0x398e[142]:
        camera[_0x398e[138]](3, r20);
        tracerIntegrator[_0x398e[41]]();
        break;
      case _0x398e[143]:
        camera[_0x398e[138]](0, r20);
        tracerIntegrator[_0x398e[41]]();
        break;
      case _0x398e[144]:
        camera[_0x398e[138]](1, r20);
        tracerIntegrator[_0x398e[41]]();
    }
    switch(dataAndEvents[_0x398e[135]]) {
      case 38:
        camera[_0x398e[145]](0, 2.5);
        tracerIntegrator[_0x398e[41]]();
        break;
      case 40:
        camera[_0x398e[145]](0, -2.5);
        tracerIntegrator[_0x398e[41]]();
        break;
      case 37:
        camera[_0x398e[145]](2.5, 0);
        tracerIntegrator[_0x398e[41]]();
        break;
      case 39:
        camera[_0x398e[145]](-2.5, 0);
        tracerIntegrator[_0x398e[41]]();
    }
  });
  initUI();
  updateStepProgram();
  var min = Date[_0x398e[146]]();
  !function render() {
    window[_0x398e[6]](function() {
      return render();
    });
    var max = Date[_0x398e[146]]();
    /** @type {number} */
    var range = max - min;
    min = max;
    /** @type {number} */
    camera[_0x398e[147]] = range;
    fps[_0x398e[148]]();
    if (!loading) {
      setStatus();
      tracerIntegrator[_0x398e[149]]();
      tracerIntegrator[_0x398e[150]]();
      Input[_0x398e[151]]()[_0x398e[148]]();
    }
  }();
}
var textures;
!function(res) {
  /**
   * @return {?}
   */
  function val() {
    return _0x6f51x2++;
  }
  /**
   * @return {undefined}
   */
  function key() {
    /** @type {number} */
    _0x6f51x2 = 0;
  }
  /** @type {number} */
  var _0x6f51x2 = 0;
  /** @type {function (): ?} */
  res[_0x398e[152]] = val;
  /** @type {function (): undefined} */
  res[_0x398e[41]] = key;
}(textures || (textures = {}));
var canvas = document[_0x398e[24]](_0x398e[153]);
/** @type {Array} */
var size = [canvas[_0x398e[154]], canvas[_0x398e[155]]];
/** @type {boolean} */
var fog = false;
var gl;
var failureMessage = _0x398e[31];
if (document[_0x398e[24]](_0x398e[156])[_0x398e[131]] = getContext(), console[_0x398e[32]](gl), !gl) {
  throw failureMessage = _0x398e[157], alert(_0x398e[158]), _0x398e[159];
}
gl[_0x398e[161]](_0x398e[160]), gl[_0x398e[161]](_0x398e[162]), getVendors(), window[_0x398e[6]] || alert(_0x398e[163]);
var globals;
!function(r) {
  /**
   * @param {?} obj
   * @return {?}
   */
  function isArray(obj) {
    var _0x6f51x2 = obj.toString();
    return _0x6f51x2[_0x398e[165]](_0x398e[164]) < 0 && (_0x6f51x2[_0x398e[165]](_0x398e[166]) < 0 && _0x6f51x2[_0x398e[165]](_0x398e[143]) < 0) ? _0x6f51x2 + _0x398e[167] : _0x6f51x2;
  }
  /**
   * @param {Array} dataAndEvents
   * @return {?}
   */
  function clone(dataAndEvents) {
    return _0x398e[168] + dataAndEvents[0] + _0x398e[0] + dataAndEvents[1] + _0x398e[169];
  }
  /**
   * @param {Array} failing_message
   * @return {?}
   */
  function report(failing_message) {
    return _0x398e[170] + failing_message[0] + _0x398e[0] + failing_message[1] + _0x398e[0] + failing_message[2] + _0x398e[169];
  }
  /** @type {number} */
  r[_0x398e[65]] = 1.000277;
  /** @type {number} */
  r[_0x398e[171]] = 1E-4;
  /** @type {number} */
  r[_0x398e[172]] = 1E-7;
  /** @type {boolean} */
  r[_0x398e[173]] = true;
  r[_0x398e[174]] = _0x398e[175];
  /** @type {function (?): ?} */
  r[_0x398e[176]] = isArray;
  /** @type {function (Array): ?} */
  r[_0x398e[177]] = clone;
  /** @type {function (Array): ?} */
  r[_0x398e[178]] = report;
}(globals || (globals = {}));
var SourceFrag = function() {
  /**
   * @param {?} dataAndEvents
   * @param {number} deepDataAndEvents
   * @return {undefined}
   */
  function clone(dataAndEvents, deepDataAndEvents) {
    if (void 0 === deepDataAndEvents) {
      /** @type {Array} */
      deepDataAndEvents = [];
    }
    /** @type {number} */
    this[_0x398e[179]] = clone[_0x398e[180]]++;
    this[_0x398e[181]] = dataAndEvents;
    /** @type {number} */
    this[_0x398e[182]] = deepDataAndEvents;
  }
  return clone[_0x398e[184]][_0x398e[183]] = function(b) {
    if (!b) {
      b = {};
    }
    var str = _0x398e[31];
    if (b[this[_0x398e[179]]]) {
      return str;
    }
    if (this[_0x398e[182]]) {
      /** @type {number} */
      var colori = 0;
      for (;colori < this[_0x398e[182]][_0x398e[3]];colori++) {
        if (!this[_0x398e[182]][colori]) {
          throw _0x398e[185];
        }
        str += this[_0x398e[182]][colori].toString(b);
      }
    }
    return str += this[_0x398e[181]], b[this[_0x398e[179]]] = true, str;
  }, clone[_0x398e[180]] = 0, clone;
}();
var sourceFrags;
!function(dataAndEvents) {
  dataAndEvents[_0x398e[186]] = new SourceFrag(_0x398e[187]);
  dataAndEvents[_0x398e[188]] = new SourceFrag(_0x398e[189]);
  dataAndEvents[_0x398e[190]] = new SourceFrag(_0x398e[191]);
  dataAndEvents[_0x398e[192]] = new SourceFrag(_0x398e[193], [dataAndEvents[_0x398e[190]]]);
  dataAndEvents[_0x398e[194]] = new SourceFrag(_0x398e[195]);
  dataAndEvents[_0x398e[196]] = new SourceFrag(_0x398e[197]);
  dataAndEvents[_0x398e[198]] = new SourceFrag(_0x398e[199]);
  dataAndEvents[_0x398e[200]] = new SourceFrag(_0x398e[201] + globals[_0x398e[171]] + _0x398e[202] + globals[_0x398e[171]] + _0x398e[203]);
  dataAndEvents[_0x398e[204]] = new SourceFrag(_0x398e[205] + globals[_0x398e[172]] + _0x398e[206]);
  dataAndEvents[_0x398e[207]] = new SourceFrag(_0x398e[208]);
  dataAndEvents[_0x398e[209]] = new SourceFrag(_0x398e[210] + globals[_0x398e[171]] + _0x398e[211]);
  dataAndEvents[_0x398e[212]] = new SourceFrag(_0x398e[213]);
  dataAndEvents[_0x398e[214]] = new SourceFrag(_0x398e[215], [dataAndEvents[_0x398e[209]], dataAndEvents[_0x398e[198]]]);
  dataAndEvents[_0x398e[216]] = new SourceFrag(_0x398e[217], [dataAndEvents[_0x398e[212]]]);
  dataAndEvents[_0x398e[218]] = new SourceFrag(_0x398e[219], [dataAndEvents[_0x398e[188]]]);
  dataAndEvents[_0x398e[220]] = new SourceFrag(_0x398e[221], [dataAndEvents[_0x398e[188]]]);
  dataAndEvents[_0x398e[222]] = new SourceFrag(_0x398e[223], [dataAndEvents[_0x398e[220]]]);
  dataAndEvents[_0x398e[224]] = new SourceFrag(_0x398e[225], [dataAndEvents[_0x398e[188]]]);
  dataAndEvents[_0x398e[226]] = new SourceFrag(_0x398e[227], [dataAndEvents[_0x398e[188]]]);
  dataAndEvents[_0x398e[228]] = new SourceFrag(_0x398e[229], [dataAndEvents[_0x398e[188]]]);
  dataAndEvents[_0x398e[230]] = new SourceFrag(_0x398e[231]);
  dataAndEvents[_0x398e[232]] = new SourceFrag(_0x398e[233]);
  dataAndEvents[_0x398e[234]] = new SourceFrag(_0x398e[235]);
  dataAndEvents[_0x398e[236]] = new SourceFrag(_0x398e[237]);
  dataAndEvents[_0x398e[238]] = new SourceFrag(_0x398e[239], [dataAndEvents[_0x398e[234]]]);
  dataAndEvents[_0x398e[240]] = new SourceFrag(_0x398e[241]);
  dataAndEvents[_0x398e[242]] = new SourceFrag(_0x398e[243], [dataAndEvents[_0x398e[240]]]);
  dataAndEvents[_0x398e[244]] = new SourceFrag(_0x398e[245], [dataAndEvents[_0x398e[240]]]);
  dataAndEvents[_0x398e[246]] = new SourceFrag(_0x398e[247], [dataAndEvents[_0x398e[240]]]);
  dataAndEvents[_0x398e[248]] = new SourceFrag(_0x398e[249], [dataAndEvents[_0x398e[240]]]);
  dataAndEvents[_0x398e[250]] = new SourceFrag(_0x398e[251], [dataAndEvents[_0x398e[240]]]);
  dataAndEvents[_0x398e[252]] = new SourceFrag(_0x398e[253], []);
  dataAndEvents[_0x398e[254]] = new SourceFrag(_0x398e[255], []);
  dataAndEvents[_0x398e[256]] = new SourceFrag(_0x398e[257], [dataAndEvents[_0x398e[254]]]);
  dataAndEvents[_0x398e[258]] = new SourceFrag(_0x398e[259], []);
  dataAndEvents[_0x398e[260]] = new SourceFrag(_0x398e[261], [dataAndEvents[_0x398e[258]]]);
  dataAndEvents[_0x398e[262]] = new SourceFrag(_0x398e[263], []);
  dataAndEvents[_0x398e[264]] = new SourceFrag(_0x398e[265], []);
}(sourceFrags || (sourceFrags = {}));
var mode;
!function(dataAndEvents) {
  dataAndEvents[dataAndEvents[_0x398e[266]] = 0] = _0x398e[266];
  dataAndEvents[dataAndEvents[_0x398e[267]] = 1] = _0x398e[267];
  dataAndEvents[dataAndEvents[_0x398e[268]] = 2] = _0x398e[268];
}(mode || (mode = {}));
var ShaderProgram = function() {
  /**
   * @return {undefined}
   */
  function _0x6f51x4() {
    this[_0x398e[269]] = {};
    this[_0x398e[270]] = {};
    /** @type {Array} */
    this[_0x398e[271]] = [];
  }
  return _0x6f51x4[_0x398e[184]][_0x398e[272]] = function(types) {
    var type;
    for (type in types) {
      type = types[type];
      this[_0x398e[270]][type] = gl[_0x398e[274]](this[_0x398e[273]], type);
    }
  }, _0x6f51x4[_0x398e[184]][_0x398e[275]] = function(types) {
    var type;
    for (type in types) {
      type = types[type];
      this[_0x398e[269]][type] = gl[_0x398e[276]](this[_0x398e[273]], type);
    }
  }, _0x6f51x4[_0x398e[184]][_0x398e[277]] = function() {
    return this[_0x398e[273]];
  }, _0x6f51x4[_0x398e[184]][_0x398e[278]] = function(dest, pdataOld, dataAndEvents) {
    var pdataCur;
    if (dataAndEvents == mode[_0x398e[266]]) {
      pdataCur = this[_0x398e[279]](dest, pdataOld);
    } else {
      if (dataAndEvents == mode[_0x398e[267]]) {
        pdataCur = this[_0x398e[280]](dest, pdataOld);
      } else {
        if (dataAndEvents == mode[_0x398e[268]]) {
          pdataCur = this[_0x398e[281]](dest, pdataOld);
        }
      }
    }
    this[_0x398e[271]][_0x398e[282]](pdataCur);
  }, _0x6f51x4[_0x398e[184]][_0x398e[283]] = function() {
    this[_0x398e[273]] = gl[_0x398e[284]]();
    /** @type {number} */
    var unlock = 0;
    for (;unlock < this[_0x398e[271]][_0x398e[3]];unlock++) {
      gl[_0x398e[285]](this[_0x398e[273]], this[_0x398e[271]][unlock]);
    }
    if (gl[_0x398e[286]](this[_0x398e[273]]), !gl[_0x398e[287]](this[_0x398e[273]], gl.LINK_STATUS)) {
      throw alert(_0x398e[159]), console[_0x398e[290]](_0x398e[288] + gl[_0x398e[289]](this[_0x398e[273]])), console[_0x398e[32]](this[_0x398e[39]]), _0x398e[291];
    }
    return true;
  }, _0x6f51x4[_0x398e[184]][_0x398e[279]] = function(message, deepDataAndEvents) {
    /** @type {XMLHttpRequest} */
    var a = new XMLHttpRequest;
    a[_0x398e[293]](_0x398e[292], message, false);
    try {
      a[_0x398e[294]]();
    } catch (d) {
      return alert(_0x398e[295] + message), console[_0x398e[32]](_0x398e[295] + message), null;
    }
    var a12 = a[_0x398e[296]];
    if (null === a12) {
      throw alert(_0x398e[297] + message + _0x398e[298]), console[_0x398e[32]](this[_0x398e[39]]), _0x398e[291];
    }
    return this[_0x398e[299]](a12, deepDataAndEvents);
  }, _0x6f51x4[_0x398e[184]][_0x398e[281]] = function(deepDataAndEvents, opt_obj2) {
    if (null === deepDataAndEvents) {
      throw alert(_0x398e[297] + deepDataAndEvents + _0x398e[298]), console[_0x398e[32]](this[_0x398e[39]]), _0x398e[291];
    }
    return this[_0x398e[299]](deepDataAndEvents, opt_obj2);
  }, _0x6f51x4[_0x398e[184]][_0x398e[280]] = function(opt_obj, deepDataAndEvents) {
    var rval;
    var r20;
    if (rval = document[_0x398e[24]](opt_obj), r20 = rval[_0x398e[301]][_0x398e[300]], null === r20) {
      throw alert(_0x398e[297] + opt_obj + _0x398e[298]), console[_0x398e[32]](this[_0x398e[39]]), _0x398e[291];
    }
    return this[_0x398e[299]](r20, deepDataAndEvents);
  }, _0x6f51x4[_0x398e[184]][_0x398e[299]] = function(deepDataAndEvents, dest) {
    var pdataCur;
    if (dest == gl[_0x398e[302]] ? this[_0x398e[303]] = deepDataAndEvents : dest == gl[_0x398e[304]] && (this[_0x398e[39]] = deepDataAndEvents), pdataCur = gl[_0x398e[305]](dest), gl[_0x398e[306]](pdataCur, deepDataAndEvents), gl[_0x398e[299]](pdataCur), !gl[_0x398e[307]](pdataCur, gl.COMPILE_STATUS)) {
      throw alert(_0x398e[295] + gl[_0x398e[308]](pdataCur)), console[_0x398e[32]](_0x398e[295] + gl[_0x398e[308]](pdataCur)), console[_0x398e[32]](this[_0x398e[39]]), _0x398e[291];
    }
    return pdataCur;
  }, _0x6f51x4[_0x398e[184]][_0x398e[309]] = function() {
    gl[_0x398e[310]](this[_0x398e[273]]);
  }, _0x6f51x4[_0x398e[184]][_0x398e[311]] = function() {
  }, _0x6f51x4;
}();
var utility;
!function(descriptor) {
  /**
   * @param {?} url
   * @return {?}
   */
  function get(url) {
    var _0x6f51x2;
    /** @type {XMLHttpRequest} */
    var request = new XMLHttpRequest;
    request[_0x398e[293]](_0x398e[292], url, false);
    try {
      request[_0x398e[294]]();
    } catch (_0x6f51x5) {
      return alert(_0x398e[312] + url), null;
    }
    return _0x6f51x2 = JSON[_0x398e[313]](request[_0x398e[296]]), delete request, _0x6f51x2;
  }
  /**
   * @param {?} key
   * @return {?}
   */
  function unlock(key) {
    var r20 = gl[_0x398e[314]]();
    /** @type {Image} */
    var res = new Image;
    return res[_0x398e[315]] = function() {
      gl[_0x398e[316]](gl.TEXTURE_2D, r20);
      gl[_0x398e[317]](gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, res);
      gl[_0x398e[318]](gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
      gl[_0x398e[318]](gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
      gl[_0x398e[318]](gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl[_0x398e[318]](gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl[_0x398e[316]](gl.TEXTURE_2D, null);
    }, res[_0x398e[319]] = function(x) {
      console[_0x398e[32]](x);
    }, res[_0x398e[181]] = key, r20;
  }
  /**
   * @param {?} vals
   * @param {?} border
   * @param {?} memo
   * @param {?} cb
   * @param {?} initial
   * @return {?}
   */
  function reduce(vals, border, memo, cb, initial) {
    /**
     * @param {?} deepDataAndEvents
     * @return {undefined}
     */
    function iterator(deepDataAndEvents) {
      if (!_0x6f51x10[deepDataAndEvents[_0x398e[320]]]) {
        /** @type {boolean} */
        _0x6f51x10[deepDataAndEvents[_0x398e[320]]] = true;
        _0x6f51xf[_0x398e[282]](deepDataAndEvents);
      }
      deepDataAndEvents[_0x398e[322]][_0x398e[321]](function(deepDataAndEvents) {
        iterator(deepDataAndEvents);
      });
      _0x6f51xe[_0x398e[282]](deepDataAndEvents);
    }
    /** @type {Array} */
    var _0x6f51xe = [];
    /** @type {Array} */
    var _0x6f51xf = [];
    var _0x6f51x10 = {};
    vals[_0x398e[321]](function(arr) {
      iterator(arr[_0x398e[62]]);
    });
    var block = _0x398e[323];
    if (memo) {
      block += memo[_0x398e[324]]();
    }
    vals[_0x398e[321]](function(dataAndEvents) {
      block += dataAndEvents[_0x398e[324]]();
    });
    _0x6f51xe[_0x398e[321]](function(dataAndEvents) {
      block += dataAndEvents[_0x398e[324]]();
    });
    block += border[_0x398e[325]]();
    block += sourceFrags[_0x398e[192]].toString() + _0x398e[326];
    block += fog ? _0x398e[327] : _0x398e[328];
    block += border[_0x398e[329]]() + _0x398e[330] + globals[_0x398e[176]](globals[_0x398e[65]]) + _0x398e[331];
    _0x6f51xf[_0x398e[321]](function(dataAndEvents) {
      block += dataAndEvents[_0x398e[332]]();
    });
    block += _0x398e[333] + cb + _0x398e[334];
    vals[_0x398e[321]](function(dataAndEvents) {
      block += dataAndEvents[_0x398e[335]]() + _0x398e[336] + dataAndEvents[_0x398e[337]] + _0x398e[338] + dataAndEvents[_0x398e[341]](_0x398e[339], _0x398e[340]) + _0x398e[342];
    });
    vals[_0x398e[321]](function(dataAndEvents) {
      block += _0x398e[343] + dataAndEvents[_0x398e[345]](dataAndEvents[_0x398e[337]] + _0x398e[344]) + _0x398e[346] + dataAndEvents[_0x398e[347]](dataAndEvents[_0x398e[337]] + _0x398e[344]) + _0x398e[348] + dataAndEvents[_0x398e[347]](dataAndEvents[_0x398e[337]] + _0x398e[344]) + _0x398e[349] + dataAndEvents[_0x398e[179]] + _0x398e[350];
    });
    block += _0x398e[351];
    vals[_0x398e[321]](function(dataAndEvents) {
      if (block += _0x398e[352] + dataAndEvents[_0x398e[179]] + _0x398e[353], dataAndEvents[_0x398e[354]]) {
        var _0x6f51x2 = dataAndEvents[_0x398e[354]](dataAndEvents[_0x398e[337]] + _0x398e[344]);
        if (_0x398e[355] !== _0x6f51x2) {
          block += _0x398e[356] + dataAndEvents[_0x398e[354]](dataAndEvents[_0x398e[337]] + _0x398e[344]) + _0x398e[342];
        }
      }
      block += _0x398e[357] + dataAndEvents[_0x398e[359]](dataAndEvents[_0x398e[337]] + _0x398e[344], _0x398e[358], _0x398e[339], _0x398e[340]) + _0x398e[342] + dataAndEvents[_0x398e[62]][_0x398e[361]](_0x398e[358], _0x398e[360], _0x398e[339], _0x398e[340]);
    });
    block += _0x398e[362] + memo[_0x398e[363]]() + _0x398e[364];
    _0x6f51xf[_0x398e[321]](function(x) {
      if (void 0 !== x[_0x398e[320]]) {
        console[_0x398e[32]](x);
        block += _0x398e[365] + x[_0x398e[320]] + _0x398e[366] + x[_0x398e[367]]();
      }
    });
    block += _0x398e[368] + globals[_0x398e[176]](cb) + _0x398e[369];
    /** @type {Array} */
    var relUrl = [sourceFrags[_0x398e[194]], initial, sourceFrags[_0x398e[250]]];
    relUrl = relUrl[_0x398e[33]](memo[_0x398e[370]]);
    relUrl = relUrl[_0x398e[33]](border[_0x398e[370]]);
    vals[_0x398e[321]](function(dataAndEvents) {
      relUrl = relUrl[_0x398e[33]](dataAndEvents[_0x398e[370]]);
    });
    _0x6f51xf[_0x398e[321]](function(dataAndEvents) {
      relUrl = relUrl[_0x398e[33]](dataAndEvents[_0x398e[370]]);
    });
    /** @type {Array} */
    var suiteView = [_0x398e[371], _0x398e[372], _0x398e[373]];
    return suiteView = suiteView[_0x398e[33]](memo[_0x398e[374]]), suiteView = suiteView[_0x398e[33]](border[_0x398e[374]]), vals[_0x398e[321]](function(dataAndEvents) {
      suiteView = suiteView[_0x398e[33]](dataAndEvents[_0x398e[374]]);
    }), _0x6f51xe[_0x398e[321]](function(dataAndEvents) {
      suiteView = suiteView[_0x398e[33]](dataAndEvents[_0x398e[374]]);
    }), forOwn((new SourceFrag(block, relUrl)).toString(), 8, suiteView);
  }
  /**
   * @param {?} opt_obj
   * @param {?} min2
   * @param {?} reject
   * @param {?} cb
   * @return {?}
   */
  function map(opt_obj, min2, reject, cb) {
    /**
     * @param {?} deepDataAndEvents
     * @return {undefined}
     */
    function scan(deepDataAndEvents) {
      if (!_0x6f51xf[deepDataAndEvents[_0x398e[320]]]) {
        /** @type {boolean} */
        _0x6f51xf[deepDataAndEvents[_0x398e[320]]] = true;
        _0x6f51xe[_0x398e[282]](deepDataAndEvents);
      }
      deepDataAndEvents[_0x398e[322]][_0x398e[321]](function(deepDataAndEvents) {
        scan(deepDataAndEvents);
      });
      _0x6f51xc[_0x398e[282]](deepDataAndEvents);
    }
    /** @type {Array} */
    var _0x6f51xc = [];
    /** @type {Array} */
    var _0x6f51xe = [];
    var _0x6f51xf = {};
    opt_obj[_0x398e[321]](function(dataAndEvents) {
      scan(dataAndEvents[_0x398e[62]]);
    });
    var block = _0x398e[375];
    if (reject) {
      block += reject[_0x398e[324]]();
    }
    opt_obj[_0x398e[321]](function(dataAndEvents) {
      block += dataAndEvents[_0x398e[324]]();
    });
    _0x6f51xc[_0x398e[321]](function(dataAndEvents) {
      block += dataAndEvents[_0x398e[324]]();
    });
    block += _0x398e[376] + min2[_0x398e[325]]() + _0x398e[377] + sourceFrags[_0x398e[192]].toString() + _0x398e[378];
    block += _0x398e[379];
    opt_obj[_0x398e[321]](function(dataAndEvents) {
      block += dataAndEvents[_0x398e[335]]() + _0x398e[336] + dataAndEvents[_0x398e[337]] + _0x398e[338] + dataAndEvents[_0x398e[341]](_0x398e[339], _0x398e[340]) + _0x398e[342];
    });
    opt_obj[_0x398e[321]](function(dataAndEvents) {
      if (!(dataAndEvents instanceof Plane)) {
        block += _0x398e[380] + dataAndEvents[_0x398e[347]](dataAndEvents[_0x398e[337]] + _0x398e[344]) + _0x398e[381];
      }
    });
    block += _0x398e[382];
    block += _0x398e[383];
    block += fog ? _0x398e[327] : _0x398e[328];
    block += _0x398e[384] + min2[_0x398e[329]]() + _0x398e[385] + globals[_0x398e[176]](globals[_0x398e[65]]) + _0x398e[386];
    _0x6f51xe[_0x398e[321]](function(dataAndEvents) {
      block += dataAndEvents[_0x398e[332]]();
    });
    block += _0x398e[387] + cb + _0x398e[388];
    opt_obj[_0x398e[321]](function(dataAndEvents) {
      block += dataAndEvents[_0x398e[335]]() + _0x398e[336] + dataAndEvents[_0x398e[337]] + _0x398e[338] + dataAndEvents[_0x398e[341]](_0x398e[339], _0x398e[340]) + _0x398e[389];
    });
    opt_obj[_0x398e[321]](function(dataAndEvents) {
      block += _0x398e[390] + dataAndEvents[_0x398e[345]](dataAndEvents[_0x398e[337]] + _0x398e[344]) + _0x398e[346] + dataAndEvents[_0x398e[347]](dataAndEvents[_0x398e[337]] + _0x398e[344]) + _0x398e[391] + dataAndEvents[_0x398e[347]](dataAndEvents[_0x398e[337]] + _0x398e[344]) + _0x398e[392] + dataAndEvents[_0x398e[179]] + _0x398e[393];
    });
    block += _0x398e[394];
    opt_obj[_0x398e[321]](function(dataAndEvents) {
      if (block += _0x398e[352] + dataAndEvents[_0x398e[179]] + _0x398e[353], dataAndEvents[_0x398e[354]]) {
        var _0x6f51x2 = dataAndEvents[_0x398e[354]](dataAndEvents[_0x398e[337]] + _0x398e[344]);
        if (_0x398e[355] !== _0x6f51x2) {
          block += _0x398e[356] + dataAndEvents[_0x398e[354]](dataAndEvents[_0x398e[337]] + _0x398e[344]) + _0x398e[389];
        }
      }
      block += _0x398e[357] + dataAndEvents[_0x398e[359]](dataAndEvents[_0x398e[337]] + _0x398e[344], _0x398e[358], _0x398e[339], _0x398e[340]) + _0x398e[395] + dataAndEvents[_0x398e[62]][_0x398e[361]](_0x398e[358], _0x398e[360], _0x398e[339], _0x398e[340]);
    });
    block += _0x398e[396] + reject[_0x398e[363]]() + _0x398e[397];
    _0x6f51xe[_0x398e[321]](function(dataAndEvents) {
      if (dataAndEvents[_0x398e[320]]) {
        block += _0x398e[365] + dataAndEvents[_0x398e[320]] + _0x398e[398] + dataAndEvents[_0x398e[367]]();
      }
    });
    block += _0x398e[399] + globals[_0x398e[176]](cb) + _0x398e[400];
    /** @type {Array} */
    var relUrl = [sourceFrags[_0x398e[194]]];
    relUrl = relUrl[_0x398e[33]](reject[_0x398e[370]]);
    relUrl = relUrl[_0x398e[33]](min2[_0x398e[370]]);
    opt_obj[_0x398e[321]](function(dataAndEvents) {
      relUrl = relUrl[_0x398e[33]](dataAndEvents[_0x398e[370]]);
    });
    _0x6f51xe[_0x398e[321]](function(dataAndEvents) {
      relUrl = relUrl[_0x398e[33]](dataAndEvents[_0x398e[370]]);
    });
    /** @type {Array} */
    var suiteView = [_0x398e[371], _0x398e[372], _0x398e[373]];
    return suiteView = suiteView[_0x398e[33]](reject[_0x398e[374]]), suiteView = suiteView[_0x398e[33]](min2[_0x398e[374]]), opt_obj[_0x398e[321]](function(dataAndEvents) {
      suiteView = suiteView[_0x398e[33]](dataAndEvents[_0x398e[374]]);
    }), _0x6f51xc[_0x398e[321]](function(dataAndEvents) {
      suiteView = suiteView[_0x398e[33]](dataAndEvents[_0x398e[374]]);
    }), forOwn((new SourceFrag(block, relUrl)).toString(), 8, suiteView);
  }
  /** @type {function (?): ?} */
  descriptor[_0x398e[401]] = get;
  /** @type {function (?): ?} */
  descriptor[_0x398e[402]] = unlock;
  /** @type {function (?, ?, ?, ?, ?): ?} */
  descriptor[_0x398e[403]] = reduce;
  /** @type {function (?, ?, ?, ?): ?} */
  descriptor[_0x398e[404]] = map;
  /**
   * @param {?} object
   * @param {number} opt_attributes
   * @param {Array} obj
   * @return {?}
   */
  var forOwn = function(object, opt_attributes, obj) {
    /** @type {Array} */
    var x = [_0x398e[405], _0x398e[406], _0x398e[407], _0x398e[408]];
    if (obj) {
      x = x[_0x398e[33]](obj);
    }
    console[_0x398e[32]](x);
    var oExtSort = new ShaderProgram;
    return oExtSort[_0x398e[278]](globals[_0x398e[174]], gl.VERTEX_SHADER, mode[_0x398e[268]]), oExtSort[_0x398e[278]](_0x398e[409] + object + _0x398e[410] + opt_attributes + _0x398e[411] + globals[_0x398e[176]](opt_attributes) + _0x398e[412], gl.FRAGMENT_SHADER, mode[_0x398e[268]]), oExtSort[_0x398e[283]](), oExtSort[_0x398e[272]]([_0x398e[413]]), oExtSort[_0x398e[275]](x), oExtSort;
  };
}(utility || (utility = {}));
var scene;
!function(res) {
  var key = function() {
    /**
     * @param {?} dataAndEvents
     * @param {?} deepDataAndEvents
     * @return {undefined}
     */
    function clone(dataAndEvents, deepDataAndEvents) {
      /** @type {Array} */
      this[_0x398e[370]] = [sourceFrags[_0x398e[218]]];
      /** @type {Array} */
      this[_0x398e[374]] = [_0x398e[55], _0x398e[414]];
      this[_0x398e[55]] = dataAndEvents;
      this[_0x398e[53]] = deepDataAndEvents;
    }
    return clone[_0x398e[184]][_0x398e[325]] = function() {
      return _0x398e[415];
    }, clone[_0x398e[184]][_0x398e[329]] = function() {
      return _0x398e[416];
    }, clone[_0x398e[184]][_0x398e[148]] = function(dataAndEvents) {
      gl[_0x398e[417]](dataAndEvents[_0x398e[269]][_0x398e[55]], this[_0x398e[55]]);
      gl[_0x398e[417]](dataAndEvents[_0x398e[269]][_0x398e[414]], this[_0x398e[53]]);
    }, clone[_0x398e[184]][_0x398e[418]] = function(dataAndEvents, deepDataAndEvents) {
      /** @type {function (Array, Array): ?} */
      var throttledUpdate = (vec3[_0x398e[419]](deepDataAndEvents[0], deepDataAndEvents[1], 1), function(dataAndEvents, deepDataAndEvents) {
        return new Float32Array([dataAndEvents[0] * deepDataAndEvents[0], dataAndEvents[1] * deepDataAndEvents[1], dataAndEvents[2], dataAndEvents[3] * deepDataAndEvents[0], dataAndEvents[4] * deepDataAndEvents[1], dataAndEvents[5], dataAndEvents[6] * deepDataAndEvents[0], dataAndEvents[7] * deepDataAndEvents[1], dataAndEvents[8]]);
      });
      return throttledUpdate(dataAndEvents, deepDataAndEvents);
    }, clone;
  }();
  res[_0x398e[420]] = key;
  var val = function() {
    /**
     * @param {?} dataAndEvents
     * @param {?} deepDataAndEvents
     * @return {undefined}
     */
    function clone(dataAndEvents, deepDataAndEvents) {
      this[_0x398e[141]] = dataAndEvents;
      this[_0x398e[421]] = deepDataAndEvents;
    }
    return clone[_0x398e[184]][_0x398e[422]] = function() {
      return this[_0x398e[141]];
    }, clone[_0x398e[184]][_0x398e[423]] = function() {
      return this[_0x398e[421]];
    }, clone[_0x398e[184]][_0x398e[424]] = function(dataAndEvents) {
      return this[_0x398e[141]] + dataAndEvents * this[_0x398e[421]];
    }, clone;
  }();
  res[_0x398e[425]] = val;
}(scene || (scene = {}));
var __extends = this && this[_0x398e[426]] || function(elems, parent) {
  /**
   * @return {undefined}
   */
  function result() {
    /** @type {Function} */
    this[_0x398e[427]] = elems;
  }
  var i;
  for (i in parent) {
    if (parent[_0x398e[428]](i)) {
      elems[i] = parent[i];
    }
  }
  elems[_0x398e[184]] = null === parent ? Object[_0x398e[429]](parent) : (result[_0x398e[184]] = parent[_0x398e[184]], new result);
};
var environments;
!function(res) {
  var Base = function() {
    /**
     * @return {undefined}
     */
    function _0x6f51x4() {
      /** @type {Array} */
      this[_0x398e[370]] = [];
    }
    return _0x6f51x4[_0x398e[184]][_0x398e[148]] = function(dataAndEvents) {
    }, _0x6f51x4;
  }();
  res[_0x398e[430]] = Base;
  var val = function(_super) {
    /**
     * @param {?} value
     * @param {?} triggerChange
     * @return {undefined}
     */
    function val(value, triggerChange) {
      _super[_0x398e[431]](this);
      this[_0x398e[181]] = value;
      /** @type {Array} */
      this[_0x398e[374]] = [];
      this[_0x398e[370]] = triggerChange;
    }
    return __extends(val, _super), val[_0x398e[184]][_0x398e[148]] = function(dataAndEvents) {
    }, val[_0x398e[184]][_0x398e[324]] = function() {
      return _0x398e[31];
    }, val[_0x398e[184]][_0x398e[363]] = function() {
      return this[_0x398e[181]];
    }, val;
  }(Base);
  res[_0x398e[432]] = val;
  var key = function(_super) {
    /**
     * @param {?} value
     * @param {?} triggerChange
     * @param {?} d
     * @return {undefined}
     */
    function val(value, triggerChange, d) {
      _super[_0x398e[431]](this);
      this[_0x398e[433]] = value;
      this[_0x398e[104]] = triggerChange;
      this[_0x398e[102]] = d;
      /** @type {Array} */
      this[_0x398e[374]] = [_0x398e[433], _0x398e[434], _0x398e[435]];
      /** @type {Array} */
      this[_0x398e[370]] = [sourceFrags[_0x398e[186]], sourceFrags[_0x398e[188]]];
    }
    return __extends(val, _super), val[_0x398e[184]][_0x398e[148]] = function(dataAndEvents) {
      gl[_0x398e[436]](gl.TEXTURE1);
      gl[_0x398e[316]](gl.TEXTURE_2D, this[_0x398e[433]]);
      gl[_0x398e[437]](dataAndEvents[_0x398e[269]][_0x398e[433]], 1);
      gl[_0x398e[417]](dataAndEvents[_0x398e[269]][_0x398e[434]], this[_0x398e[102]]);
      gl[_0x398e[417]](dataAndEvents[_0x398e[269]][_0x398e[435]], this[_0x398e[104]]);
    }, val[_0x398e[184]][_0x398e[324]] = function() {
      return _0x398e[438];
    }, val[_0x398e[184]][_0x398e[363]] = function() {
      var _0x6f51x4;
      var _0x6f51x2 = _0x398e[439];
      return _0x6f51x4 = _0x398e[440] + _0x6f51x2 + _0x398e[441], _0x398e[442] + _0x6f51x4 + _0x398e[443];
    }, val;
  }(Base);
  res[_0x398e[444]] = key;
}(environments || (environments = {}));
var Camera = function() {
  /**
   * @param {number} dataAndEvents
   * @param {number} deepDataAndEvents
   * @param {number} events
   * @param {number} keepData
   * @return {undefined}
   */
  function clone(dataAndEvents, deepDataAndEvents, events, keepData) {
    if (void 0 === dataAndEvents) {
      dataAndEvents = vec3[_0x398e[419]](0, 0, 0);
    }
    if (void 0 === deepDataAndEvents) {
      deepDataAndEvents = vec3[_0x398e[419]](0, 1, 0);
    }
    if (void 0 === events) {
      /** @type {number} */
      events = -90;
    }
    if (void 0 === keepData) {
      /** @type {number} */
      keepData = 0;
    }
    /** @type {number} */
    this[_0x398e[445]] = 0.5;
    /** @type {number} */
    this[_0x398e[446]] = 0.25;
    this[_0x398e[447]] = mat4[_0x398e[429]]();
    this[_0x398e[448]] = mat4[_0x398e[429]]();
    this[_0x398e[449]] = vec3[_0x398e[419]](0, 0, -1);
    /** @type {number} */
    this[_0x398e[450]] = dataAndEvents;
    /** @type {number} */
    this[_0x398e[451]] = deepDataAndEvents;
    /** @type {number} */
    this[_0x398e[452]] = events;
    /** @type {number} */
    this[_0x398e[453]] = keepData;
    this[_0x398e[454]] = vec3[_0x398e[429]]();
    this[_0x398e[455]] = vec3[_0x398e[429]]();
    this[_0x398e[456]]();
  }
  return clone[_0x398e[184]][_0x398e[457]] = function() {
    return this[_0x398e[450]];
  }, clone[_0x398e[184]][_0x398e[138]] = function(dataAndEvents, deepDataAndEvents) {
    /** @type {number} */
    var r20 = this[_0x398e[445]] * this[_0x398e[147]];
    if (0 == dataAndEvents) {
      this[_0x398e[450]] = vec3[_0x398e[458]](this[_0x398e[450]], this[_0x398e[450]], this[_0x398e[449]], r20);
    } else {
      if (1 == dataAndEvents) {
        this[_0x398e[450]] = vec3[_0x398e[458]](this[_0x398e[450]], this[_0x398e[450]], this[_0x398e[449]], -r20);
      } else {
        if (2 == dataAndEvents) {
          this[_0x398e[450]] = vec3[_0x398e[458]](this[_0x398e[450]], this[_0x398e[450]], this[_0x398e[454]], -r20);
        } else {
          if (3 == dataAndEvents) {
            this[_0x398e[450]] = vec3[_0x398e[458]](this[_0x398e[450]], this[_0x398e[450]], this[_0x398e[454]], r20);
          } else {
            if (4 == dataAndEvents) {
              this[_0x398e[450]] = vec3[_0x398e[458]](this[_0x398e[450]], this[_0x398e[450]], this[_0x398e[455]], r20);
            } else {
              if (5 == dataAndEvents) {
                this[_0x398e[450]] = vec3[_0x398e[458]](this[_0x398e[450]], this[_0x398e[450]], this[_0x398e[455]], -r20);
              }
            }
          }
        }
      }
    }
  }, clone[_0x398e[184]][_0x398e[145]] = function(dataAndEvents, deepDataAndEvents) {
    this[_0x398e[452]] += dataAndEvents;
    this[_0x398e[453]] += deepDataAndEvents;
    if (this[_0x398e[453]] > 89) {
      /** @type {number} */
      this[_0x398e[453]] = 89;
    }
    if (this[_0x398e[453]] < -89) {
      /** @type {number} */
      this[_0x398e[453]] = -89;
    }
    this[_0x398e[456]]();
  }, clone[_0x398e[184]][_0x398e[456]] = function() {
    var r20 = vec3[_0x398e[419]](Math[_0x398e[460]](glMatrix[_0x398e[459]](this[_0x398e[452]])) * Math[_0x398e[460]](glMatrix[_0x398e[459]](this[_0x398e[453]])), Math[_0x398e[461]](glMatrix[_0x398e[459]](this[_0x398e[453]])), Math[_0x398e[461]](glMatrix[_0x398e[459]](this[_0x398e[452]])) * Math[_0x398e[460]](glMatrix[_0x398e[459]](this[_0x398e[453]])));
    this[_0x398e[449]] = vec3[_0x398e[462]](this[_0x398e[449]], r20);
    this[_0x398e[454]] = vec3[_0x398e[463]](this[_0x398e[454]], this[_0x398e[449]], this[_0x398e[451]]);
    this[_0x398e[454]] = vec3[_0x398e[462]](this[_0x398e[454]], this[_0x398e[454]]);
    this[_0x398e[455]] = vec3[_0x398e[463]](this[_0x398e[455]], this[_0x398e[454]], this[_0x398e[449]]);
    this[_0x398e[455]] = vec3[_0x398e[462]](this[_0x398e[455]], this[_0x398e[455]]);
  }, clone[_0x398e[184]][_0x398e[464]] = function() {
    var r20 = vec3[_0x398e[429]]();
    return this[_0x398e[447]] = mat4[_0x398e[466]](this[_0x398e[447]], this[_0x398e[450]], vec3[_0x398e[465]](r20, this[_0x398e[450]], this[_0x398e[449]]), this[_0x398e[455]]), this[_0x398e[447]];
  }, clone[_0x398e[184]][_0x398e[467]] = function() {
    return this[_0x398e[448]] = mat4[_0x398e[468]](this[_0x398e[448]], 45, size[0] / size[1], 0.1, 100), this[_0x398e[448]];
  }, clone;
}();
var Renderable = function() {
  /**
   * @param {?} dataAndEvents
   * @param {?} deepDataAndEvents
   * @param {?} events
   * @return {undefined}
   */
  function clone(dataAndEvents, deepDataAndEvents, events) {
    /** @type {Array} */
    this[_0x398e[370]] = [];
    /** @type {number} */
    this[_0x398e[179]] = clone[_0x398e[469]]++;
    this[_0x398e[337]] = dataAndEvents + this[_0x398e[179]];
    this[_0x398e[470]] = deepDataAndEvents;
    this[_0x398e[62]] = events;
  }
  return clone[_0x398e[13]] = function() {
    /** @type {number} */
    this[_0x398e[469]] = 1;
  }, clone[_0x398e[469]] = 1, clone;
}();
__extends = this && this[_0x398e[426]] || function(elems, parent) {
  /**
   * @return {undefined}
   */
  function result() {
    /** @type {Function} */
    this[_0x398e[427]] = elems;
  }
  var i;
  for (i in parent) {
    if (parent[_0x398e[428]](i)) {
      elems[i] = parent[i];
    }
  }
  elems[_0x398e[184]] = null === parent ? Object[_0x398e[429]](parent) : (result[_0x398e[184]] = parent[_0x398e[184]], new result);
};
var Box = function(_super) {
  /**
   * @param {?} value
   * @param {?} triggerChange
   * @param {string} d
   * @param {?} s
   * @param {?} newValue
   * @return {undefined}
   */
  function val(value, triggerChange, d, s, newValue) {
    _super[_0x398e[431]](this, _0x398e[471], d, s);
    this[_0x398e[472]] = newValue;
    this[_0x398e[473]] = value;
    this[_0x398e[474]] = triggerChange;
    this[_0x398e[475]] = this[_0x398e[337]] + _0x398e[473];
    this[_0x398e[476]] = this[_0x398e[337]] + _0x398e[474];
    this[_0x398e[477]] = this[_0x398e[337]] + _0x398e[25];
    this[_0x398e[478]] = this[_0x398e[337]] + _0x398e[479];
    /** @type {Array} */
    this[_0x398e[370]] = [sourceFrags[_0x398e[198]], sourceFrags[_0x398e[200]]];
    /** @type {Array} */
    this[_0x398e[374]] = d ? [this[_0x398e[475]], this[_0x398e[476]]] : [];
  }
  return __extends(val, _super), val[_0x398e[184]][_0x398e[148]] = function(dataAndEvents) {
    if (this[_0x398e[470]]) {
      gl[_0x398e[480]](dataAndEvents[_0x398e[269]][this[_0x398e[475]]], vec3[_0x398e[419]](this[_0x398e[473]][0], this[_0x398e[473]][1], this[_0x398e[473]][2]));
      gl[_0x398e[480]](dataAndEvents[_0x398e[269]][this[_0x398e[476]]], vec3[_0x398e[419]](this[_0x398e[474]][0], this[_0x398e[474]][1], this[_0x398e[474]][2]));
    }
  }, val[_0x398e[184]][_0x398e[324]] = function() {
    return this[_0x398e[470]] ? _0x398e[481] + this[_0x398e[475]] + _0x398e[482] + this[_0x398e[476]] + _0x398e[483] + this[_0x398e[477]] + _0x398e[484] + this[_0x398e[476]] + _0x398e[485] + this[_0x398e[475]] + _0x398e[486] + this[_0x398e[478]] + _0x398e[484] + this[_0x398e[476]] + _0x398e[487] + this[_0x398e[475]] + _0x398e[488] : _0x398e[31];
  }, val[_0x398e[184]][_0x398e[335]] = function() {
    return _0x398e[489];
  }, val[_0x398e[184]][_0x398e[341]] = function(dataAndEvents, deepDataAndEvents) {
    var _0x6f51x3 = this[_0x398e[470]] ? this[_0x398e[475]] : globals[_0x398e[178]](this[_0x398e[473]]);
    var _0x6f51x5 = this[_0x398e[470]] ? this[_0x398e[476]] : globals[_0x398e[178]](this[_0x398e[474]]);
    return _0x398e[490] + _0x6f51x3 + _0x398e[491] + _0x6f51x5 + _0x398e[491] + dataAndEvents + _0x398e[491] + deepDataAndEvents + _0x398e[492];
  }, val[_0x398e[184]][_0x398e[345]] = function(dataAndEvents) {
    return this[_0x398e[472]] ? _0x398e[493] + dataAndEvents + _0x398e[494] + globals[_0x398e[172]] + _0x398e[346] + dataAndEvents + _0x398e[495] + dataAndEvents + _0x398e[496] : _0x398e[493] + dataAndEvents + _0x398e[497] + globals[_0x398e[172]] + _0x398e[346] + dataAndEvents + _0x398e[495] + dataAndEvents + _0x398e[496];
  }, val[_0x398e[184]][_0x398e[347]] = function(dataAndEvents) {
    return this[_0x398e[472]] ? _0x398e[493] + dataAndEvents + _0x398e[497] + globals[_0x398e[172]] + _0x398e[498] + dataAndEvents + _0x398e[499] + dataAndEvents + _0x398e[496] : dataAndEvents + _0x398e[500];
  }, val[_0x398e[184]][_0x398e[354]] = function(dataAndEvents) {
    return this[_0x398e[472]] ? _0x398e[493] + dataAndEvents + _0x398e[501] : _0x398e[355];
  }, val[_0x398e[184]][_0x398e[359]] = function(dataAndEvents, deepDataAndEvents, ignoreMethodDoesntExist, textAlt) {
    var fromIndex = vec3[_0x398e[429]]();
    fromIndex = vec3[_0x398e[465]](fromIndex, this[_0x398e[474]], this[_0x398e[473]]);
    fromIndex = vec3[_0x398e[419]](fromIndex[0] / 2, fromIndex[1] / 2, fromIndex[2] / 2);
    var otherElement = vec3[_0x398e[429]]();
    otherElement = vec3[_0x398e[502]](otherElement, this[_0x398e[474]], this[_0x398e[473]]);
    otherElement = vec3[_0x398e[419]](otherElement[0] / 2, otherElement[1] / 2, otherElement[2] / 2);
    var index = globals[_0x398e[178]](fromIndex);
    var otherElementRect = globals[_0x398e[178]](otherElement);
    return this[_0x398e[472]] ? _0x398e[503] + dataAndEvents + _0x398e[504] + index + _0x398e[491] + otherElementRect + _0x398e[491] + deepDataAndEvents + _0x398e[505] : _0x398e[506] + index + _0x398e[491] + otherElementRect + _0x398e[491] + deepDataAndEvents + _0x398e[507];
  }, val[_0x398e[184]][_0x398e[508]] = function(dataAndEvents) {
    var _0x6f51x2 = vec3[_0x398e[419]]((this[_0x398e[473]][0] - dataAndEvents[_0x398e[422]]()[0]) / dataAndEvents[_0x398e[423]]()[0], (this[_0x398e[473]][1] - dataAndEvents[_0x398e[422]]()[1]) / dataAndEvents[_0x398e[423]]()[1], (this[_0x398e[473]][2] - dataAndEvents[_0x398e[422]]()[2]) / dataAndEvents[_0x398e[423]]()[2]);
    var _0x6f51x3 = vec3[_0x398e[419]]((this[_0x398e[474]][0] - dataAndEvents[_0x398e[422]]()[0]) / dataAndEvents[_0x398e[423]]()[0], (this[_0x398e[474]][1] - dataAndEvents[_0x398e[422]]()[1]) / dataAndEvents[_0x398e[423]]()[1], (this[_0x398e[474]][2] - dataAndEvents[_0x398e[422]]()[2]) / dataAndEvents[_0x398e[423]]()[2]);
    var arr = vec3[_0x398e[419]](Math[_0x398e[473]](_0x6f51x2[0], _0x6f51x3[0]), Math[_0x398e[473]](_0x6f51x2[1], _0x6f51x3[1]), Math[_0x398e[473]](_0x6f51x2[2], _0x6f51x3[2]));
    var _0x6f51xc = vec3[_0x398e[419]](Math[_0x398e[474]](_0x6f51x2[0], _0x6f51x3[0]), Math[_0x398e[474]](_0x6f51x2[1], _0x6f51x3[1]), Math[_0x398e[474]](_0x6f51x2[2], _0x6f51x3[2]));
    var xDelta = Math[_0x398e[474]](Math[_0x398e[474]](arr[0], arr[1]), arr[2]);
    var yDelta = Math[_0x398e[473]](Math[_0x398e[473]](_0x6f51xc[0], _0x6f51xc[1]), _0x6f51xc[2]);
    return xDelta >= yDelta || 1E-5 > xDelta ? Number[_0x398e[509]] : xDelta;
  }, val;
}(Renderable);
__extends = this && this[_0x398e[426]] || function(elems, parent) {
  /**
   * @return {undefined}
   */
  function result() {
    /** @type {Function} */
    this[_0x398e[427]] = elems;
  }
  var i;
  for (i in parent) {
    if (parent[_0x398e[428]](i)) {
      elems[i] = parent[i];
    }
  }
  elems[_0x398e[184]] = null === parent ? Object[_0x398e[429]](parent) : (result[_0x398e[184]] = parent[_0x398e[184]], new result);
};
var Sphere = function(_super) {
  /**
   * @param {?} triggerChange
   * @param {?} s
   * @param {?} value
   * @param {?} d
   * @param {?} newValue
   * @return {undefined}
   */
  function val(triggerChange, s, value, d, newValue) {
    _super[_0x398e[431]](this, _0x398e[510], value, d);
    this[_0x398e[472]] = newValue;
    this[_0x398e[25]] = triggerChange;
    this[_0x398e[511]] = s;
    this[_0x398e[512]] = this[_0x398e[337]] + _0x398e[511];
    this[_0x398e[477]] = this[_0x398e[337]] + _0x398e[25];
    /** @type {Array} */
    this[_0x398e[370]] = [sourceFrags[_0x398e[204]], sourceFrags[_0x398e[207]]];
    this[_0x398e[56]]();
  }
  return __extends(val, _super), val[_0x398e[184]][_0x398e[148]] = function(dataAndEvents) {
    if (this[_0x398e[470]]) {
      gl[_0x398e[480]](dataAndEvents[_0x398e[269]][this[_0x398e[477]]], this[_0x398e[25]]);
      gl[_0x398e[417]](dataAndEvents[_0x398e[269]][this[_0x398e[512]]], this[_0x398e[511]]);
    }
  }, val[_0x398e[184]][_0x398e[56]] = function() {
    if (this[_0x398e[470]]) {
      this[_0x398e[513]] = this[_0x398e[512]];
      this[_0x398e[514]] = this[_0x398e[477]];
    } else {
      this[_0x398e[514]] = globals[_0x398e[178]](this[_0x398e[25]]);
      this[_0x398e[513]] = globals[_0x398e[176]](this[_0x398e[511]]);
      this[_0x398e[514]] = _0x398e[515] + this[_0x398e[514]] + _0x398e[516];
    }
    /** @type {Array} */
    this[_0x398e[374]] = this[_0x398e[470]] ? [this[_0x398e[512]], this[_0x398e[477]]] : [];
  }, val[_0x398e[184]][_0x398e[324]] = function() {
    return this[_0x398e[470]] ? _0x398e[481] + this[_0x398e[477]] + _0x398e[517] + this[_0x398e[512]] + _0x398e[342] : _0x398e[31];
  }, val[_0x398e[184]][_0x398e[335]] = function() {
    return _0x398e[489];
  }, val[_0x398e[184]][_0x398e[341]] = function(dataAndEvents, deepDataAndEvents) {
    return _0x398e[518] + this[_0x398e[514]] + _0x398e[491] + this[_0x398e[513]] + _0x398e[491] + dataAndEvents + _0x398e[491] + deepDataAndEvents + _0x398e[492];
  }, val[_0x398e[184]][_0x398e[345]] = function(dataAndEvents) {
    return this[_0x398e[472]] ? _0x398e[493] + dataAndEvents + _0x398e[494] + globals[_0x398e[172]] + _0x398e[346] + dataAndEvents + _0x398e[495] + dataAndEvents + _0x398e[496] : _0x398e[493] + dataAndEvents + _0x398e[497] + globals[_0x398e[172]] + _0x398e[346] + dataAndEvents + _0x398e[495] + dataAndEvents + _0x398e[496];
  }, val[_0x398e[184]][_0x398e[347]] = function(dataAndEvents) {
    return this[_0x398e[472]] ? _0x398e[493] + dataAndEvents + _0x398e[497] + globals[_0x398e[172]] + _0x398e[498] + dataAndEvents + _0x398e[499] + dataAndEvents + _0x398e[496] : dataAndEvents + _0x398e[500];
  }, val[_0x398e[184]][_0x398e[354]] = function(dataAndEvents) {
    return this[_0x398e[472]] ? _0x398e[493] + dataAndEvents + _0x398e[501] : _0x398e[355];
  }, val[_0x398e[184]][_0x398e[359]] = function(dataAndEvents, deepDataAndEvents, ignoreMethodDoesntExist, textAlt) {
    return this[_0x398e[472]] ? _0x398e[503] + dataAndEvents + _0x398e[519] + this[_0x398e[514]] + _0x398e[491] + this[_0x398e[513]] + _0x398e[491] + deepDataAndEvents + _0x398e[505] : _0x398e[520] + this[_0x398e[514]] + _0x398e[491] + this[_0x398e[513]] + _0x398e[491] + deepDataAndEvents + _0x398e[507];
  }, val[_0x398e[184]][_0x398e[508]] = function(dataAndEvents) {
    var r20 = vec3[_0x398e[521]](dataAndEvents[_0x398e[422]](), dataAndEvents[_0x398e[422]](), this[_0x398e[25]]);
    var rate = vec3[_0x398e[522]](dataAndEvents[_0x398e[423]](), dataAndEvents[_0x398e[423]]());
    /** @type {number} */
    var a01 = 2 * vec3[_0x398e[522]](r20, dataAndEvents[_0x398e[423]]());
    /** @type {number} */
    var a11 = vec3[_0x398e[522]](r20, r20) - this[_0x398e[511]] * this[_0x398e[511]];
    /** @type {number} */
    var value = a01 * a01 - 4 * rate * a11;
    if (value > 1E-5) {
      var amount = Math[_0x398e[523]](value);
      /** @type {number} */
      var _0x6f51xf = (-amount - a01) / (2 * rate);
      if (_0x6f51xf > 1E-5) {
        return _0x6f51xf;
      }
      /** @type {number} */
      var _0x6f51x10 = (amount - a01) / (2 * rate);
      if (_0x6f51x10 > 1E-5) {
        return _0x6f51x10;
      }
    }
    return Number[_0x398e[509]];
  }, val;
}(Renderable);
__extends = this && this[_0x398e[426]] || function(elems, parent) {
  /**
   * @return {undefined}
   */
  function result() {
    /** @type {Function} */
    this[_0x398e[427]] = elems;
  }
  var i;
  for (i in parent) {
    if (parent[_0x398e[428]](i)) {
      elems[i] = parent[i];
    }
  }
  elems[_0x398e[184]] = null === parent ? Object[_0x398e[429]](parent) : (result[_0x398e[184]] = parent[_0x398e[184]], new result);
};
var Light = function(_super) {
  /**
   * @param {?} value
   * @param {?} d
   * @param {number} triggerChange
   * @return {undefined}
   */
  function val(value, d, triggerChange) {
    _super[_0x398e[431]](this, value, d, true, new mat.Emit(new mat.Absorb, vec3[_0x398e[419]](25, 25, 25), true, triggerChange), false);
  }
  return __extends(val, _super), val[_0x398e[184]][_0x398e[30]] = function() {
    return this[_0x398e[62]][_0x398e[524]];
  }, val[_0x398e[184]][_0x398e[59]] = function(dataAndEvents) {
    this[_0x398e[62]][_0x398e[59]](vec3[_0x398e[419]](dataAndEvents[_0x398e[525]], dataAndEvents[_0x398e[526]], dataAndEvents[_0x398e[527]]));
  }, val[_0x398e[184]][_0x398e[63]] = function(m13) {
    /** @type {Float32Array} */
    this[_0x398e[62]][_0x398e[61]] = new Float32Array([m13, m13, m13]);
  }, val[_0x398e[184]][_0x398e[148]] = function(dataAndEvents) {
    gl[_0x398e[480]](dataAndEvents[_0x398e[269]][this[_0x398e[477]]], this[_0x398e[25]]);
    gl[_0x398e[417]](dataAndEvents[_0x398e[269]][this[_0x398e[512]]], this[_0x398e[511]]);
  }, val[_0x398e[184]][_0x398e[56]] = function() {
    this[_0x398e[513]] = this[_0x398e[512]];
    this[_0x398e[514]] = this[_0x398e[477]];
    /** @type {Array} */
    this[_0x398e[374]] = [this[_0x398e[512]], this[_0x398e[477]]];
  }, val;
}(Sphere);
__extends = this && this[_0x398e[426]] || function(elems, parent) {
  /**
   * @return {undefined}
   */
  function result() {
    /** @type {Function} */
    this[_0x398e[427]] = elems;
  }
  var i;
  for (i in parent) {
    if (parent[_0x398e[428]](i)) {
      elems[i] = parent[i];
    }
  }
  elems[_0x398e[184]] = null === parent ? Object[_0x398e[429]](parent) : (result[_0x398e[184]] = parent[_0x398e[184]], new result);
};
var Plane = function(_super) {
  /**
   * @param {?} triggerChange
   * @param {?} s
   * @param {string} value
   * @param {?} d
   * @return {undefined}
   */
  function val(triggerChange, s, value, d) {
    _super[_0x398e[431]](this, _0x398e[528], value, d);
    this[_0x398e[360]] = triggerChange;
    this[_0x398e[529]] = s;
    this[_0x398e[530]] = this[_0x398e[337]] + _0x398e[360];
    this[_0x398e[531]] = this[_0x398e[337]] + _0x398e[529];
    /** @type {Array} */
    this[_0x398e[370]] = [sourceFrags[_0x398e[196]]];
    /** @type {Array} */
    this[_0x398e[374]] = value ? [this[_0x398e[530]], this[_0x398e[531]]] : [];
  }
  return __extends(val, _super), val[_0x398e[184]][_0x398e[354]] = function(dataAndEvents) {
    return false;
  }, val[_0x398e[184]][_0x398e[148]] = function(dataAndEvents) {
    if (this[_0x398e[470]]) {
      gl[_0x398e[480]](dataAndEvents[_0x398e[269]][this[_0x398e[530]]], this[_0x398e[360]]);
      gl[_0x398e[417]](dataAndEvents[_0x398e[269]][this[_0x398e[531]]], this[_0x398e[529]]);
    }
  }, val[_0x398e[184]][_0x398e[324]] = function() {
    return this[_0x398e[470]] ? _0x398e[481] + this[_0x398e[530]] + _0x398e[517] + this[_0x398e[531]] + _0x398e[342] : _0x398e[532] + this[_0x398e[530]] + _0x398e[533] + globals[_0x398e[178]](this[_0x398e[360]]) + _0x398e[534] + this[_0x398e[531]] + _0x398e[533] + globals[_0x398e[176]](this[_0x398e[529]]) + _0x398e[342];
  }, val[_0x398e[184]][_0x398e[335]] = function() {
    return _0x398e[535];
  }, val[_0x398e[184]][_0x398e[341]] = function(dataAndEvents, deepDataAndEvents) {
    return _0x398e[536] + this[_0x398e[530]] + _0x398e[491] + this[_0x398e[531]] + _0x398e[491] + dataAndEvents + _0x398e[491] + deepDataAndEvents + _0x398e[492];
  }, val[_0x398e[184]][_0x398e[345]] = function(dataAndEvents) {
    return _0x398e[493] + dataAndEvents + _0x398e[537] + globals[_0x398e[172]] + _0x398e[169];
  }, val[_0x398e[184]][_0x398e[347]] = function(dataAndEvents) {
    return dataAndEvents;
  }, val[_0x398e[184]][_0x398e[359]] = function(dataAndEvents, deepDataAndEvents, ignoreMethodDoesntExist, textAlt) {
    return this[_0x398e[530]];
  }, val[_0x398e[184]][_0x398e[508]] = function(dataAndEvents) {
    /** @type {number} */
    var v = (this[_0x398e[529]] - vec3[_0x398e[522]](this[_0x398e[360]], dataAndEvents[_0x398e[422]]())) / vec3[_0x398e[522]](this[_0x398e[360]], dataAndEvents[_0x398e[423]]());
    return v > globals[_0x398e[172]] ? v : Number[_0x398e[509]];
  }, val;
}(Renderable);
__extends = this && this[_0x398e[426]] || function(elems, parent) {
  /**
   * @return {undefined}
   */
  function result() {
    /** @type {Function} */
    this[_0x398e[427]] = elems;
  }
  var i;
  for (i in parent) {
    if (parent[_0x398e[428]](i)) {
      elems[i] = parent[i];
    }
  }
  elems[_0x398e[184]] = null === parent ? Object[_0x398e[429]](parent) : (result[_0x398e[184]] = parent[_0x398e[184]], new result);
};
var Triangle = function(_super) {
  /**
   * @param {?} triggerChange
   * @param {?} s
   * @param {?} newValue
   * @param {?} value
   * @param {?} d
   * @return {undefined}
   */
  function val(triggerChange, s, newValue, value, d) {
    _super[_0x398e[431]](this, _0x398e[538], value, d);
    this[_0x398e[539]] = triggerChange;
    this[_0x398e[540]] = s;
    this[_0x398e[541]] = newValue;
    /** @type {Array} */
    this[_0x398e[370]] = [sourceFrags[_0x398e[209]], sourceFrags[_0x398e[212]]];
  }
  return __extends(val, _super), val[_0x398e[184]][_0x398e[148]] = function(dataAndEvents) {
  }, val[_0x398e[184]][_0x398e[324]] = function() {
    return _0x398e[31];
  }, val[_0x398e[184]][_0x398e[335]] = function() {
    return _0x398e[535];
  }, val[_0x398e[184]][_0x398e[341]] = function(dataAndEvents, deepDataAndEvents) {
    return _0x398e[542] + dataAndEvents + _0x398e[491] + deepDataAndEvents + _0x398e[491] + globals[_0x398e[178]](this[_0x398e[539]]) + _0x398e[491] + globals[_0x398e[178]](this[_0x398e[540]]) + _0x398e[491] + globals[_0x398e[178]](this[_0x398e[541]]) + _0x398e[543];
  }, val[_0x398e[184]][_0x398e[345]] = function(dataAndEvents) {
    return _0x398e[493] + dataAndEvents + _0x398e[537] + globals[_0x398e[172]] + _0x398e[169];
  }, val[_0x398e[184]][_0x398e[347]] = function(dataAndEvents) {
    return dataAndEvents;
  }, val[_0x398e[184]][_0x398e[354]] = function(dataAndEvents) {
    return false;
  }, val[_0x398e[184]][_0x398e[359]] = function(dataAndEvents, deepDataAndEvents, ignoreMethodDoesntExist, textAlt) {
    return _0x398e[544] + globals[_0x398e[178]](this[_0x398e[539]]) + _0x398e[491] + globals[_0x398e[178]](this[_0x398e[540]]) + _0x398e[491] + globals[_0x398e[178]](this[_0x398e[541]]) + _0x398e[169];
  }, val[_0x398e[184]][_0x398e[508]] = function(dataAndEvents) {
  }, val;
}(Renderable);
__extends = this && this[_0x398e[426]] || function(elems, parent) {
  /**
   * @return {undefined}
   */
  function result() {
    /** @type {Function} */
    this[_0x398e[427]] = elems;
  }
  var i;
  for (i in parent) {
    if (parent[_0x398e[428]](i)) {
      elems[i] = parent[i];
    }
  }
  elems[_0x398e[184]] = null === parent ? Object[_0x398e[429]](parent) : (result[_0x398e[184]] = parent[_0x398e[184]], new result);
};
var Mesh = function(_super) {
  /**
   * @param {?} value
   * @param {?} d
   * @param {?} triggerChange
   * @return {undefined}
   */
  function val(value, d, triggerChange) {
    _super[_0x398e[431]](this, _0x398e[545], d, triggerChange);
    /** @type {Array} */
    this[_0x398e[546]] = [-26.1086, 40.2493, 14.4406, -8.9987, 9.0651, 21.6143, 17.9654, 22.1507, 14.8697, -26.1086, 40.2493, 14.4406, 17.9654, 22.1507, 14.8697, 8.9944, 17.7971, -10, -26.1086, 40.2493, 14.4406, 8.9944, 17.7971, -10, -5.4071, 10.8081, -2.7681, -26.1086, 40.2493, 14.4406, -5.4071, 10.8081, -2.7681, -8.9987, 9.0651, 21.6143, -8.9987, 9.0651, 21.6143, -0.0022, 13.4311, -0, 17.9654, 22.1507, 14.8697, 8.9944, 17.7971, -10, -5.4071, 10.8081, -2.7681];
    /** @type {Array} */
    this[_0x398e[547]] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 13, 12, 14, 13, 15, 15, 13, 16];
    this[_0x398e[548]]();
    /** @type {Array} */
    this[_0x398e[374]] = [_0x398e[549], _0x398e[550], _0x398e[551], _0x398e[552]];
    this[_0x398e[553]] = gl[_0x398e[554]]();
    gl[_0x398e[555]](gl.ARRAY_BUFFER, this[_0x398e[553]]);
    gl[_0x398e[556]](gl.ARRAY_BUFFER, new Float32Array(this[_0x398e[546]]), gl.STATIC_DRAW);
    this[_0x398e[557]] = gl[_0x398e[554]]();
    gl[_0x398e[555]](gl.ELEMENT_ARRAY_BUFFER, this[_0x398e[557]]);
    gl[_0x398e[556]](gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(this[_0x398e[547]]), gl.STATIC_DRAW);
    this[_0x398e[558]] = gl[_0x398e[314]]();
    gl[_0x398e[316]](gl.TEXTURE_2D, this[_0x398e[558]]);
    gl[_0x398e[318]](gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl[_0x398e[318]](gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl[_0x398e[318]](gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl[_0x398e[318]](gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    /** @type {number} */
    var x = this[_0x398e[546]][_0x398e[3]] / 3;
    console[_0x398e[32]](x);
    /** @type {Float32Array} */
    var temp = new Float32Array(4 * x);
    /** @type {number} */
    var pos = 0;
    /** @type {number} */
    var i = 0;
    for (;i < this[_0x398e[546]][_0x398e[3]];i += 3) {
      temp[pos++] = this[_0x398e[546]][i];
      temp[pos++] = this[_0x398e[546]][i + 1];
      temp[pos++] = this[_0x398e[546]][i + 2];
      /** @type {number} */
      temp[pos++] = 0;
    }
    gl[_0x398e[317]](gl.TEXTURE_2D, 0, gl.RGBA, x, 1, 0, gl.RGBA, gl.FLOAT, temp);
    delete temp;
    this[_0x398e[559]] = gl[_0x398e[314]]();
    gl[_0x398e[316]](gl.TEXTURE_2D, this[_0x398e[559]]);
    gl[_0x398e[318]](gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl[_0x398e[318]](gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl[_0x398e[318]](gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl[_0x398e[318]](gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    console[_0x398e[32]](this[_0x398e[547]][_0x398e[3]] / 4);
    /** @type {Uint16Array} */
    var ints = new Uint16Array(this[_0x398e[547]][_0x398e[3]]);
    /** @type {number} */
    pos = 0;
    /** @type {number} */
    i = 0;
    for (;i < this[_0x398e[547]][_0x398e[3]];i += 4) {
      ints[pos++] = this[_0x398e[547]][i];
      ints[pos++] = this[_0x398e[547]][i + 1];
      ints[pos++] = this[_0x398e[547]][i + 2];
      ints[pos++] = this[_0x398e[547]][i + 3];
    }
    gl[_0x398e[317]](gl.TEXTURE_2D, 0, gl.RGBA16UI, this[_0x398e[547]][_0x398e[3]] / 4, 1, 0, gl.RGBA_INTEGER, gl.UNSIGNED_SHORT, ints);
    delete ints;
    /** @type {number} */
    var l = this[_0x398e[547]][_0x398e[3]] / 4;
    var _0x6f51x12 = _0x398e[560] + this[_0x398e[547]][_0x398e[3]] / 3 + _0x398e[561];
    var _0x6f51x13 = _0x398e[31];
    /** @type {number} */
    i = 0;
    for (;i < this[_0x398e[547]][_0x398e[3]];i += 3) {
      _0x6f51x13 += _0x398e[562] + this[_0x398e[547]][i] + _0x398e[491] + this[_0x398e[547]][i + 1] + _0x398e[491] + this[_0x398e[547]][i + 2] + _0x398e[563];
    }
    _0x6f51x13 = _0x6f51x13[_0x398e[564]](0, _0x6f51x13[_0x398e[3]] - 1);
    _0x6f51x12 += _0x398e[336] + _0x6f51x13 + _0x398e[565] + globals[_0x398e[178]](this[_0x398e[566]]) + _0x398e[567] + globals[_0x398e[178]](this[_0x398e[568]]) + _0x398e[569] + globals[_0x398e[172]] + _0x398e[570];
    /** @type {number} */
    i = 0;
    for (;l > i;i++) {
      _0x6f51x12 += _0x398e[571] + l + _0x398e[572] + x + _0x398e[573] + x + _0x398e[574] + x + _0x398e[575];
    }
    _0x6f51x12 += _0x398e[576];
    sourceFrags[_0x398e[577]] = new SourceFrag(_0x6f51x12, [sourceFrags[_0x398e[198]], sourceFrags[_0x398e[209]]]);
    /** @type {Array} */
    this[_0x398e[370]] = [sourceFrags[_0x398e[577]], sourceFrags[_0x398e[216]]];
  }
  return __extends(val, _super), val[_0x398e[184]][_0x398e[548]] = function() {
    var toRight = Number[_0x398e[578]];
    var minX = Number[_0x398e[578]];
    var timestamp = Number[_0x398e[578]];
    var totalPage = Number[_0x398e[579]];
    var dim = Number[_0x398e[579]];
    var y1 = Number[_0x398e[579]];
    /** @type {number} */
    var unlock = 0;
    for (;unlock < this[_0x398e[546]][_0x398e[3]];unlock += 3) {
      var maxDim = vec3[_0x398e[419]](this[_0x398e[546]][unlock], this[_0x398e[546]][unlock + 1], this[_0x398e[546]][unlock + 2]);
      toRight = toRight < maxDim[0] ? toRight : maxDim[0];
      minX = minX < maxDim[1] ? minX : maxDim[1];
      timestamp = timestamp < maxDim[2] ? timestamp : maxDim[2];
      totalPage = totalPage > maxDim[0] ? totalPage : maxDim[0];
      dim = dim > maxDim[1] ? dim : maxDim[1];
      y1 = y1 > maxDim[2] ? y1 : maxDim[2];
    }
    this[_0x398e[566]] = vec3[_0x398e[419]](toRight, minX, timestamp);
    this[_0x398e[568]] = vec3[_0x398e[419]](totalPage, dim, y1);
  }, val[_0x398e[184]][_0x398e[324]] = function() {
    return _0x398e[31];
  }, val[_0x398e[184]][_0x398e[148]] = function(dataAndEvents) {
    gl[_0x398e[436]](gl.TEXTURE5);
    gl[_0x398e[316]](gl.TEXTURE_2D, this[_0x398e[558]]);
    gl[_0x398e[437]](dataAndEvents[_0x398e[269]][_0x398e[549]], 5);
    gl[_0x398e[580]](dataAndEvents[_0x398e[269]].VERTEX_TEX_SIZE, this[_0x398e[546]][_0x398e[3]]);
    gl[_0x398e[436]](gl.TEXTURE6);
    gl[_0x398e[316]](gl.TEXTURE_2D, this[_0x398e[559]]);
    gl[_0x398e[437]](dataAndEvents[_0x398e[269]][_0x398e[550]], 6);
    gl[_0x398e[580]](dataAndEvents[_0x398e[269]].TRIANGLE_TEX_SIZE, this[_0x398e[547]][_0x398e[3]]);
  }, val[_0x398e[184]][_0x398e[354]] = function(dataAndEvents) {
    return false;
  }, val[_0x398e[184]][_0x398e[335]] = function() {
    return _0x398e[489];
  }, val[_0x398e[184]][_0x398e[345]] = function(dataAndEvents) {
    return _0x398e[493] + dataAndEvents + _0x398e[497] + globals[_0x398e[172]] + _0x398e[169];
  }, val[_0x398e[184]][_0x398e[347]] = function(dataAndEvents) {
    return dataAndEvents + _0x398e[500];
  }, val[_0x398e[184]][_0x398e[341]] = function(dataAndEvents, deepDataAndEvents) {
    return _0x398e[581] + dataAndEvents + _0x398e[491] + deepDataAndEvents + _0x398e[543];
  }, val[_0x398e[184]][_0x398e[359]] = function(deepDataAndEvents, dataAndEvents, ignoreMethodDoesntExist, textAlt) {
    return _0x398e[582] + dataAndEvents + _0x398e[583];
  }, val[_0x398e[184]][_0x398e[508]] = function(dataAndEvents) {
  }, val;
}(Renderable);
var SceneLoader = function() {
  /**
   * @param {?} error
   * @return {?}
   */
  function init(error) {
    /** @type {Array} */
    this[_0x398e[14]] = [];
    /** @type {Array} */
    this[_0x398e[15]] = [];
    /** @type {Array} */
    this[_0x398e[16]] = [];
    /** @type {Array} */
    this[_0x398e[17]] = [];
    /** @type {Array} */
    this[_0x398e[18]] = [];
    /** @type {Array} */
    this[_0x398e[34]] = [];
    /** @type {XMLHttpRequest} */
    var req = new XMLHttpRequest;
    req[_0x398e[293]](_0x398e[292], error, false);
    try {
      req[_0x398e[294]]();
    } catch (c) {
      return alert(_0x398e[295] + error), null;
    }
    /** @type {DOMParser} */
    var parser = new DOMParser;
    this[_0x398e[584]] = parser[_0x398e[586]](req[_0x398e[296]], _0x398e[585]);
    this[_0x398e[587]] = this[_0x398e[584]][_0x398e[588]](_0x398e[587])[_0x398e[127]](0);
    this[_0x398e[589]]();
    this[_0x398e[590]]();
    var r20 = this[_0x398e[587]][_0x398e[588]](_0x398e[591])[_0x398e[127]](0);
    this[_0x398e[592]](r20);
    this[_0x398e[593]](r20);
    this[_0x398e[594]](r20);
    this[_0x398e[595]](r20);
    this[_0x398e[596]](r20);
    this[_0x398e[597]](r20);
  }
  return init[_0x398e[184]][_0x398e[598]] = function(dataAndEvents, rgb) {
    if (void 0 === rgb) {
      rgb = _0x398e[336];
    }
    var components = dataAndEvents[_0x398e[1]](rgb);
    var length = components[_0x398e[3]];
    /** @type {Array} */
    var values = new Array(length);
    /** @type {number} */
    var index = 0;
    for (;length > index;index++) {
      /** @type {number} */
      values[index] = parseFloat(components[index]);
    }
    return values;
  }, init[_0x398e[184]][_0x398e[599]] = function(dataAndEvents) {
    var result;
    switch(dataAndEvents[_0x398e[604]]) {
      case _0x398e[602]:
        var restoreScript = dataAndEvents[_0x398e[126]](_0x398e[600]);
        var rreturn = dataAndEvents[_0x398e[126]](_0x398e[601]);
        var udataCur = dataAndEvents[_0x398e[126]](_0x398e[360]);
        result = new mat.Textured(utility[_0x398e[402]](restoreScript), utility[_0x398e[402]](rreturn), utility[_0x398e[402]](udataCur));
        break;
      case _0x398e[608]:
        if (_0x398e[603] == dataAndEvents[_0x398e[605]][_0x398e[604]]) {
          var _0x6f51xd = dataAndEvents[_0x398e[588]](_0x398e[603])[_0x398e[127]](0);
          var r20 = this[_0x398e[607]](_0x6f51xd[_0x398e[126]](_0x398e[606]));
          var top = this[_0x398e[598]](_0x6f51xd[_0x398e[126]](_0x398e[524]));
          result = new mat.Emit(new mat.Absorb, vec3[_0x398e[419]](r20, r20, r20), false, top);
        }
        break;
      case _0x398e[609]:
        result = new mat.Diffuse(globals[_0x398e[173]]);
        break;
      case _0x398e[610]:
        result = new mat.Attenuate(this[_0x398e[599]](dataAndEvents[_0x398e[605]]), this[_0x398e[598]](dataAndEvents[_0x398e[126]](_0x398e[524])), false);
        break;
      case _0x398e[615]:
        if (2 == dataAndEvents[_0x398e[611]]) {
          var x = this[_0x398e[599]](dataAndEvents[_0x398e[612]][_0x398e[127]](1));
          console[_0x398e[32]](x);
          var message = this[_0x398e[599]](dataAndEvents[_0x398e[612]][_0x398e[127]](3));
          console[_0x398e[32]](message);
          var params = this[_0x398e[607]](dataAndEvents[_0x398e[126]](_0x398e[613]));
          var hour = this[_0x398e[607]](dataAndEvents[_0x398e[126]](_0x398e[614]));
          result = new mat.FresnelComposite(x, message, params, hour);
        }
        break;
      case _0x398e[619]:
        if (dataAndEvents[_0x398e[126]](_0x398e[64]) && (dataAndEvents[_0x398e[126]](_0x398e[616]) && dataAndEvents[_0x398e[126]](_0x398e[617]))) {
          /** @type {boolean} */
          var right = 1 == this[_0x398e[607]](dataAndEvents[_0x398e[126]](_0x398e[617]));
          var _0x6f51x15 = this[_0x398e[607]](dataAndEvents[_0x398e[126]](_0x398e[64]));
          result = new mat.Metal([_0x6f51x15, _0x6f51x15], this[_0x398e[607]](dataAndEvents[_0x398e[126]](_0x398e[616])), right);
        } else {
          if (dataAndEvents[_0x398e[126]](_0x398e[618]) && (dataAndEvents[_0x398e[126]](_0x398e[616]) && dataAndEvents[_0x398e[126]](_0x398e[617]))) {
            /** @type {boolean} */
            right = 1 == this[_0x398e[607]](dataAndEvents[_0x398e[126]](_0x398e[617]));
            result = new mat.Metal(this[_0x398e[598]](dataAndEvents[_0x398e[126]](_0x398e[618])), this[_0x398e[607]](dataAndEvents[_0x398e[126]](_0x398e[616])), right);
          }
        }
        if (dataAndEvents[_0x398e[126]](_0x398e[524])) {
          result[_0x398e[524]] = this[_0x398e[598]](dataAndEvents[_0x398e[126]](_0x398e[524]));
        }
        break;
      case _0x398e[622]:
        if (dataAndEvents[_0x398e[621]][_0x398e[620]](_0x398e[524])) {
          result = new mat.SmoothDielectric(this[_0x398e[607]](dataAndEvents[_0x398e[126]](_0x398e[64])));
          this[_0x398e[598]](dataAndEvents[_0x398e[126]](_0x398e[524]));
        } else {
          result = new mat.SmoothDielectric(this[_0x398e[607]](dataAndEvents[_0x398e[126]](_0x398e[64])));
        }
        if (dataAndEvents[_0x398e[126]](_0x398e[524])) {
          result[_0x398e[524]] = this[_0x398e[598]](dataAndEvents[_0x398e[126]](_0x398e[524]));
        }
        break;
      case _0x398e[624]:
        result = new mat.PhongSpecular(this[_0x398e[607]](dataAndEvents[_0x398e[126]](_0x398e[623])), false);
        break;
      case _0x398e[627]:
        var uvs = this[_0x398e[598]](dataAndEvents[_0x398e[126]](_0x398e[625]));
        var length = this[_0x398e[598]](dataAndEvents[_0x398e[126]](_0x398e[626]));
        result = new mat.ChessTextured(new Float32Array(uvs), new Float32Array(length));
        break;
      case _0x398e[628]:
        result = new mat[_0x398e[628]];
        break;
      case _0x398e[629]:
        result = new mat[_0x398e[629]];
        break;
      case _0x398e[630]:
        uvs = this[_0x398e[598]](dataAndEvents[_0x398e[126]](_0x398e[625]));
        length = this[_0x398e[598]](dataAndEvents[_0x398e[126]](_0x398e[626]));
        result = uvs ? length ? new mat.WorleyTexture(new Float32Array(uvs), new Float32Array(length)) : new mat.WorleyTexture(new Float32Array(uvs)) : new mat[_0x398e[630]];
        break;
      case _0x398e[633]:
        if (dataAndEvents[_0x398e[126]](_0x398e[631])) {
          if (dataAndEvents[_0x398e[126]](_0x398e[632])) {
            result = new mat.Transmit(this[_0x398e[607]](dataAndEvents[_0x398e[126]](_0x398e[631])), this[_0x398e[607]](dataAndEvents[_0x398e[126]](_0x398e[632])));
          }
        }
        break;
      case _0x398e[634]:
        result = new mat[_0x398e[634]];
        break;
      default:
        console[_0x398e[32]](dataAndEvents[_0x398e[604]]);
    }
    return result;
  }, init[_0x398e[184]][_0x398e[607]] = function(sValue) {
    return parseFloat(sValue);
  }, init[_0x398e[184]][_0x398e[589]] = function() {
    var _0x6f51x4 = this[_0x398e[587]][_0x398e[588]](_0x398e[635])[_0x398e[127]](0);
    /** @type {Float32Array} */
    var out = new Float32Array(this[_0x398e[598]](_0x6f51x4[_0x398e[621]][_0x398e[620]](_0x398e[450])[_0x398e[22]]));
    /** @type {Float32Array} */
    var array = new Float32Array(this[_0x398e[598]](_0x6f51x4[_0x398e[621]][_0x398e[620]](_0x398e[455])[_0x398e[22]]));
    var width = this[_0x398e[607]](_0x6f51x4[_0x398e[621]][_0x398e[620]](_0x398e[452])[_0x398e[22]]);
    var height = this[_0x398e[607]](_0x6f51x4[_0x398e[621]][_0x398e[620]](_0x398e[453])[_0x398e[22]]);
    this[_0x398e[28]] = new Camera(out, array, width, height);
  }, init[_0x398e[184]][_0x398e[590]] = function() {
    var _0x6f51x4 = this[_0x398e[587]][_0x398e[588]](_0x398e[636])[_0x398e[127]](0);
    this[_0x398e[21]] = new Light(this[_0x398e[598]](_0x6f51x4[_0x398e[621]][_0x398e[620]](_0x398e[450])[_0x398e[22]]), this[_0x398e[607]](_0x6f51x4[_0x398e[621]][_0x398e[620]](_0x398e[511])[_0x398e[22]]), this[_0x398e[598]](_0x6f51x4[_0x398e[621]][_0x398e[620]](_0x398e[524])[_0x398e[22]]));
    /** @type {number} */
    var num2 = parseInt(_0x6f51x4[_0x398e[126]](_0x398e[606]));
    this[_0x398e[21]][_0x398e[63]](num2);
  }, init[_0x398e[184]][_0x398e[592]] = function(dataAndEvents) {
    var ws = dataAndEvents[_0x398e[588]](_0x398e[637]);
    /** @type {number} */
    var y = 0;
    var x = ws[_0x398e[3]];
    for (;x > y;y++) {
      var _0x6f51xc = this[_0x398e[598]](ws[y][_0x398e[621]][_0x398e[620]](_0x398e[25])[_0x398e[22]]);
      var optional_timeout_ = this[_0x398e[607]](ws[y][_0x398e[621]][_0x398e[620]](_0x398e[511])[_0x398e[22]]);
      var optional_timeoutMessage_ = this[_0x398e[599]](ws[y][_0x398e[605]]);
      var waitsForFunc = new Sphere(_0x6f51xc, optional_timeout_, false, optional_timeoutMessage_, true);
      if (optional_timeoutMessage_) {
        this[_0x398e[14]][_0x398e[282]](waitsForFunc);
      }
    }
  }, init[_0x398e[184]][_0x398e[593]] = function(dataAndEvents) {
    var vals = dataAndEvents[_0x398e[588]](_0x398e[638]);
    /** @type {number} */
    var i = 0;
    var val = vals[_0x398e[3]];
    for (;val > i;i++) {
      var min = this[_0x398e[598]](vals[i][_0x398e[126]](_0x398e[473]));
      var timeout = this[_0x398e[598]](vals[i][_0x398e[126]](_0x398e[474]));
      var frameYsize = this[_0x398e[599]](vals[i][_0x398e[605]]);
      var waitsFunc = new Box(min, timeout, false, frameYsize, true);
      if (frameYsize) {
        this[_0x398e[15]][_0x398e[282]](waitsFunc);
      }
    }
  }, init[_0x398e[184]][_0x398e[594]] = function(dataAndEvents) {
    var vals = dataAndEvents[_0x398e[588]](_0x398e[639]);
    /** @type {number} */
    var i = 0;
    var val = vals[_0x398e[3]];
    for (;val > i;i++) {
      var audiolet = this[_0x398e[598]](vals[i][_0x398e[126]](_0x398e[539]));
      var fbr = this[_0x398e[598]](vals[i][_0x398e[126]](_0x398e[540]));
      var ffl = this[_0x398e[598]](vals[i][_0x398e[126]](_0x398e[541]));
      var _0x6f51xf = this[_0x398e[599]](vals[i][_0x398e[605]]);
      var triangle = new Triangle(audiolet, fbr, ffl, false, _0x6f51xf);
      if (_0x6f51xf) {
        this[_0x398e[16]][_0x398e[282]](triangle);
      }
    }
  }, init[_0x398e[184]][_0x398e[595]] = function(dataAndEvents) {
    var params = dataAndEvents[_0x398e[588]](_0x398e[640]);
    /** @type {number} */
    var cacheParam = 0;
    var param = params[_0x398e[3]];
    for (;param > cacheParam;cacheParam++) {
      var geometry = params[cacheParam][_0x398e[126]](_0x398e[181]);
      var scaleFactor = this[_0x398e[599]](params[cacheParam][_0x398e[605]]);
      var mesh = new Mesh(geometry, false, scaleFactor);
      if (scaleFactor) {
        this[_0x398e[17]][_0x398e[282]](mesh);
      }
    }
  }, init[_0x398e[184]][_0x398e[596]] = function(dataAndEvents) {
    var clonedXProps = dataAndEvents[_0x398e[588]](_0x398e[641]);
    /** @type {number} */
    var b = 0;
    var a = clonedXProps[_0x398e[3]];
    for (;a > b;b++) {
      var str = this[_0x398e[598]](clonedXProps[b][_0x398e[126]](_0x398e[360]));
      var timeout = this[_0x398e[598]](clonedXProps[b][_0x398e[126]](_0x398e[529]));
      var color = this[_0x398e[599]](clonedXProps[b][_0x398e[605]]);
      var waitsFunc = new Plane(str, timeout, false, color);
      if (color) {
        this[_0x398e[18]][_0x398e[282]](waitsFunc);
      }
    }
  }, init[_0x398e[184]][_0x398e[597]] = function(dataAndEvents) {
    var args = dataAndEvents[_0x398e[588]](_0x398e[636]);
    /** @type {number} */
    var idx = 0;
    var pageY = args[_0x398e[3]];
    for (;pageY > idx;idx++) {
      var next = args[idx];
      var r20 = new Light(this[_0x398e[598]](next[_0x398e[621]][_0x398e[620]](_0x398e[450])[_0x398e[22]]), this[_0x398e[607]](next[_0x398e[621]][_0x398e[620]](_0x398e[511])[_0x398e[22]]), this[_0x398e[598]](next[_0x398e[621]][_0x398e[620]](_0x398e[524])[_0x398e[22]]));
      /** @type {number} */
      var num2 = parseInt(next[_0x398e[126]](_0x398e[606]));
      r20[_0x398e[63]](num2);
      this[_0x398e[34]][_0x398e[282]](r20);
    }
  }, init;
}();
__extends = this && this[_0x398e[426]] || function(elems, parent) {
  /**
   * @return {undefined}
   */
  function result() {
    /** @type {Function} */
    this[_0x398e[427]] = elems;
  }
  var i;
  for (i in parent) {
    if (parent[_0x398e[428]](i)) {
      elems[i] = parent[i];
    }
  }
  elems[_0x398e[184]] = null === parent ? Object[_0x398e[429]](parent) : (result[_0x398e[184]] = parent[_0x398e[184]], new result);
};
var mat;
!function(results) {
  var Base = function() {
    /**
     * @return {undefined}
     */
    function _0x6f51x4() {
      /** @type {number} */
      this[_0x398e[179]] = _0x6f51x4[_0x398e[642]]++;
      /** @type {Array} */
      this[_0x398e[374]] = [];
      /** @type {Array} */
      this[_0x398e[322]] = [];
      /** @type {Array} */
      this[_0x398e[370]] = [];
    }
    return _0x6f51x4[_0x398e[184]][_0x398e[148]] = function(dataAndEvents) {
    }, _0x6f51x4[_0x398e[184]][_0x398e[324]] = function() {
      return _0x398e[31];
    }, _0x6f51x4[_0x398e[184]][_0x398e[332]] = function() {
      return _0x398e[31];
    }, _0x6f51x4[_0x398e[184]][_0x398e[367]] = function() {
      return _0x398e[31];
    }, _0x6f51x4[_0x398e[642]] = 1, _0x6f51x4[_0x398e[643]] = 1, _0x6f51x4;
  }();
  results[_0x398e[644]] = Base;
  var result = function(_super) {
    /**
     * @return {undefined}
     */
    function destElements() {
      _super[_0x398e[431]](this);
      /** @type {number} */
      this[_0x398e[320]] = Base[_0x398e[643]]++;
    }
    return __extends(destElements, _super), destElements[_0x398e[184]][_0x398e[361]] = function(dataAndEvents, deepDataAndEvents, ignoreMethodDoesntExist, textAlt) {
      return _0x398e[645] + this[_0x398e[320]] + _0x398e[389];
    }, destElements[_0x398e[184]][_0x398e[367]] = function() {
      return _0x398e[646];
    }, destElements;
  }(Base);
  results[_0x398e[603]] = result;
  var content = function(_super) {
    /**
     * @return {undefined}
     */
    function destElements() {
      _super[_0x398e[431]](this);
      /** @type {number} */
      this[_0x398e[320]] = Base[_0x398e[643]]++;
    }
    return __extends(destElements, _super), destElements[_0x398e[184]][_0x398e[361]] = function(dataAndEvents, deepDataAndEvents, ignoreMethodDoesntExist, textAlt) {
      return _0x398e[645] + this[_0x398e[320]] + _0x398e[389];
    }, destElements[_0x398e[184]][_0x398e[367]] = function() {
      return _0x398e[647] + globals[_0x398e[171]] + _0x398e[648];
    }, destElements;
  }(Base);
  results[_0x398e[634]] = content;
  var component = function(_super) {
    /**
     * @param {?} value
     * @param {?} triggerChange
     * @return {undefined}
     */
    function val(value, triggerChange) {
      _super[_0x398e[431]](this);
      /** @type {number} */
      this[_0x398e[320]] = Base[_0x398e[643]]++;
      this[_0x398e[613]] = value;
      this[_0x398e[614]] = triggerChange;
    }
    return __extends(val, _super), val[_0x398e[184]][_0x398e[361]] = function(dataAndEvents, deepDataAndEvents, ignoreMethodDoesntExist, textAlt) {
      return _0x398e[649] + this[_0x398e[179]] + _0x398e[650] + this[_0x398e[613]] + _0x398e[491] + this[_0x398e[614]] + _0x398e[651] + this[_0x398e[179]] + _0x398e[652] + this[_0x398e[179]] + _0x398e[653] + this[_0x398e[320]] + _0x398e[389];
    }, val[_0x398e[184]][_0x398e[332]] = function() {
      return _0x398e[654] + this[_0x398e[179]] + _0x398e[389];
    }, val[_0x398e[184]][_0x398e[367]] = function() {
      return _0x398e[655] + this[_0x398e[179]] + _0x398e[656] + this[_0x398e[179]] + _0x398e[657] + globals[_0x398e[171]] + _0x398e[648];
    }, val;
  }(Base);
  results[_0x398e[633]] = component;
  var Collection = function(_super) {
    /**
     * @param {?} dataAndEvents
     * @return {undefined}
     */
    function destElements(dataAndEvents) {
      _super[_0x398e[431]](this);
      this[_0x398e[658]] = dataAndEvents;
      /** @type {number} */
      this[_0x398e[320]] = Base[_0x398e[643]]++;
      /** @type {Array} */
      this[_0x398e[370]] = [sourceFrags[_0x398e[224]], sourceFrags[_0x398e[230]], sourceFrags[_0x398e[188]]];
    }
    return __extends(destElements, _super), destElements[_0x398e[184]][_0x398e[361]] = function(dataAndEvents, deepDataAndEvents, ignoreMethodDoesntExist, textAlt) {
      return _0x398e[645] + this[_0x398e[320]] + _0x398e[389];
    }, destElements[_0x398e[184]][_0x398e[367]] = function() {
      return this[_0x398e[658]] + _0x398e[659];
    }, destElements;
  }(Base);
  results[_0x398e[660]] = Collection;
  var collection = function(_super) {
    /**
     * @return {undefined}
     */
    function destElements() {
      _super[_0x398e[431]](this, _0x398e[661]);
    }
    return __extends(destElements, _super), destElements;
  }(Collection);
  results[_0x398e[629]] = collection;
  var v = function(_super) {
    /**
     * @param {?} value
     * @param {?} triggerChange
     * @return {undefined}
     */
    function val(value, triggerChange) {
      var r20 = _0x398e[662] + value + _0x398e[663] + triggerChange + _0x398e[664];
      _super[_0x398e[431]](this, r20);
    }
    return __extends(val, _super), val;
  }(Collection);
  results[_0x398e[665]] = v;
  var value = function(_super) {
    /**
     * @return {undefined}
     */
    function destElements() {
      var r20 = _0x398e[666];
      _super[_0x398e[431]](this, r20);
      this[_0x398e[370]] = this[_0x398e[370]][_0x398e[33]]([sourceFrags[_0x398e[252]], sourceFrags[_0x398e[260]], sourceFrags[_0x398e[186]], sourceFrags[_0x398e[188]]]);
    }
    return __extends(destElements, _super), destElements;
  }(Collection);
  results[_0x398e[628]] = value;
  var args = function(_super) {
    /**
     * @param {number} value
     * @param {number} d
     * @param {number} a
     * @return {undefined}
     */
    function val(value, d, a) {
      if (void 0 === value) {
        value = vec3[_0x398e[419]](0, 0, 0);
      }
      if (void 0 === d) {
        d = vec3[_0x398e[419]](1, 1, 1);
      }
      if (void 0 === a) {
        /** @type {number} */
        a = 0.4;
      }
      if (0 > a || a > 1) {
        /** @type {number} */
        a = 0.4;
      }
      var r20 = _0x398e[667] + globals[_0x398e[178]](value) + _0x398e[491] + globals[_0x398e[176]](a) + _0x398e[668] + globals[_0x398e[178]](d) + _0x398e[669];
      _super[_0x398e[431]](this, r20);
      this[_0x398e[370]] = this[_0x398e[370]][_0x398e[33]]([sourceFrags[_0x398e[252]], sourceFrags[_0x398e[256]], sourceFrags[_0x398e[186]], sourceFrags[_0x398e[188]]]);
    }
    return __extends(val, _super), val;
  }(Collection);
  results[_0x398e[630]] = args;
  var mapped = function(_super) {
    /**
     * @param {?} value
     * @param {?} triggerChange
     * @param {?} d
     * @return {undefined}
     */
    function val(value, triggerChange, d) {
      _super[_0x398e[431]](this);
      this[_0x398e[670]] = value;
      this[_0x398e[671]] = triggerChange;
      this[_0x398e[672]] = d;
      /** @type {Array} */
      this[_0x398e[374]] = [_0x398e[673], _0x398e[674], _0x398e[675]];
      /** @type {number} */
      this[_0x398e[320]] = Base[_0x398e[643]]++;
      /** @type {Array} */
      this[_0x398e[370]] = [sourceFrags[_0x398e[234]], sourceFrags[_0x398e[224]], sourceFrags[_0x398e[230]], sourceFrags[_0x398e[188]]];
    }
    return __extends(val, _super), val[_0x398e[184]][_0x398e[148]] = function(dataAndEvents) {
      gl[_0x398e[436]](gl.TEXTURE2);
      gl[_0x398e[316]](gl.TEXTURE_2D, this[_0x398e[670]]);
      gl[_0x398e[437]](dataAndEvents[_0x398e[269]][_0x398e[673]], 2);
      gl[_0x398e[436]](gl.TEXTURE3);
      gl[_0x398e[316]](gl.TEXTURE_2D, this[_0x398e[671]]);
      gl[_0x398e[437]](dataAndEvents[_0x398e[269]][_0x398e[674]], 3);
      gl[_0x398e[436]](gl.TEXTURE4);
      gl[_0x398e[316]](gl.TEXTURE_2D, this[_0x398e[672]]);
      gl[_0x398e[437]](dataAndEvents[_0x398e[269]][_0x398e[675]], 4);
    }, val[_0x398e[184]][_0x398e[324]] = function() {
      return _0x398e[676];
    }, val[_0x398e[184]][_0x398e[361]] = function(dataAndEvents, deepDataAndEvents, ignoreMethodDoesntExist, textAlt) {
      return _0x398e[645] + this[_0x398e[320]] + _0x398e[389];
    }, val[_0x398e[184]][_0x398e[367]] = function() {
      return _0x398e[677] + globals[_0x398e[171]] + _0x398e[678];
    }, val;
  }(Base);
  results[_0x398e[602]] = mapped;
  var chunk = function(_super) {
    /**
     * @param {?} value
     * @param {number} triggerChange
     * @return {undefined}
     */
    function val(value, triggerChange) {
      if (void 0 === triggerChange) {
        triggerChange = vec3[_0x398e[419]](1, 1, 1);
      }
      _super[_0x398e[431]](this);
      /** @type {number} */
      this[_0x398e[320]] = Base[_0x398e[643]]++;
      this[_0x398e[64]] = value;
      /** @type {number} */
      this[_0x398e[524]] = triggerChange;
      /** @type {Array} */
      this[_0x398e[370]] = [sourceFrags[_0x398e[234]], sourceFrags[_0x398e[232]], sourceFrags[_0x398e[262]]];
    }
    return __extends(val, _super), val[_0x398e[184]][_0x398e[361]] = function(dataAndEvents, deepDataAndEvents, ignoreMethodDoesntExist, textAlt) {
      return _0x398e[679] + globals[_0x398e[176]](globals[_0x398e[65]]) + _0x398e[680] + this[_0x398e[64]] + _0x398e[681] + this[_0x398e[320]] + _0x398e[389];
    }, val[_0x398e[184]][_0x398e[367]] = function() {
      return _0x398e[682] + globals[_0x398e[172]] + _0x398e[683] + globals[_0x398e[178]](this[_0x398e[524]]) + _0x398e[684] + globals[_0x398e[171]] + _0x398e[648];
    }, val;
  }(Base);
  results[_0x398e[622]] = chunk;
  var needsEncrypt = function(_super) {
    /**
     * @param {?} value
     * @param {string} triggerChange
     * @return {undefined}
     */
    function val(value, triggerChange) {
      _super[_0x398e[431]](this);
      this[_0x398e[623]] = value;
      this[_0x398e[685]] = _0x398e[686] + this[_0x398e[179]];
      /** @type {number} */
      this[_0x398e[320]] = Base[_0x398e[643]]++;
      /** @type {Array} */
      this[_0x398e[374]] = triggerChange ? [this[_0x398e[685]]] : [];
      /** @type {string} */
      this[_0x398e[470]] = triggerChange;
      /** @type {Array} */
      this[_0x398e[370]] = [sourceFrags[_0x398e[188]], sourceFrags[_0x398e[230]], sourceFrags[_0x398e[224]]];
    }
    return __extends(val, _super), val[_0x398e[184]][_0x398e[148]] = function(dataAndEvents) {
      if (this[_0x398e[470]]) {
        gl[_0x398e[417]](dataAndEvents[_0x398e[269]][this[_0x398e[685]]], this[_0x398e[623]]);
      }
    }, val[_0x398e[184]][_0x398e[324]] = function() {
      return this[_0x398e[470]] ? _0x398e[687] + this[_0x398e[685]] + _0x398e[389] : _0x398e[688] + this[_0x398e[685]] + _0x398e[533] + globals[_0x398e[176]](this[_0x398e[623]]) + _0x398e[389];
    }, val[_0x398e[184]][_0x398e[361]] = function(dataAndEvents, deepDataAndEvents, ignoreMethodDoesntExist, textAlt) {
      return _0x398e[689] + this[_0x398e[179]] + _0x398e[533] + this[_0x398e[685]] + _0x398e[681] + this[_0x398e[320]] + _0x398e[389];
    }, val[_0x398e[184]][_0x398e[332]] = function() {
      return _0x398e[690] + this[_0x398e[179]] + _0x398e[389];
    }, val[_0x398e[184]][_0x398e[367]] = function() {
      var _0x6f51x4 = _0x398e[691] + this[_0x398e[179]] + _0x398e[692] + this[_0x398e[179]] + _0x398e[693] + globals[_0x398e[171]] + _0x398e[648];
      return _0x6f51x4;
    }, val;
  }(Base);
  results[_0x398e[624]] = needsEncrypt;
  var oResult = function(_super) {
    /**
     * @param {?} triggerChange
     * @param {number} d
     * @param {?} value
     * @param {number} s
     * @return {undefined}
     */
    function val(triggerChange, d, value, s) {
      if (void 0 === s) {
        s = vec3[_0x398e[419]](1, 1, 1);
      }
      _super[_0x398e[431]](this);
      this[_0x398e[64]] = triggerChange;
      /** @type {number} */
      this[_0x398e[320]] = Base[_0x398e[643]]++;
      /** @type {number} */
      this[_0x398e[524]] = s;
      this[_0x398e[616]] = void 0 == d ? 0.8 : d;
      this[_0x398e[694]] = value ? _0x398e[224] : _0x398e[222];
      /** @type {Array} */
      this[_0x398e[370]] = [sourceFrags[_0x398e[236]], sourceFrags[this[_0x398e[694]]]];
    }
    return __extends(val, _super), val[_0x398e[184]][_0x398e[361]] = function(dataAndEvents, deepDataAndEvents, ignoreMethodDoesntExist, textAlt) {
      return _0x398e[695] + this[_0x398e[179]] + _0x398e[533] + globals[_0x398e[177]](this[_0x398e[64]]) + _0x398e[681] + this[_0x398e[320]] + _0x398e[696] + this[_0x398e[179]] + _0x398e[533] + globals[_0x398e[176]](this[_0x398e[616]]) + _0x398e[389];
    }, val[_0x398e[184]][_0x398e[332]] = function() {
      return _0x398e[697] + this[_0x398e[179]] + _0x398e[698] + this[_0x398e[179]] + _0x398e[699];
    }, val[_0x398e[184]][_0x398e[367]] = function() {
      return _0x398e[700] + this[_0x398e[179]] + _0x398e[701] + this[_0x398e[179]] + _0x398e[702] + globals[_0x398e[178]](this[_0x398e[524]]) + _0x398e[703] + this[_0x398e[179]] + _0x398e[704] + this[_0x398e[694]] + _0x398e[705] + globals[_0x398e[171]] + _0x398e[648];
    }, val;
  }(Base);
  results[_0x398e[619]] = oResult;
  var path = function(_super) {
    /**
     * @param {?} value
     * @param {?} triggerChange
     * @param {?} d
     * @return {undefined}
     */
    function val(value, triggerChange, d) {
      _super[_0x398e[431]](this);
      /** @type {number} */
      this[_0x398e[320]] = Base[_0x398e[643]]++;
      this[_0x398e[706]] = value;
      this[_0x398e[707]] = triggerChange;
      /** @type {Array} */
      this[_0x398e[322]] = [value, triggerChange];
      this[_0x398e[370]] = d;
    }
    return __extends(val, _super), val[_0x398e[184]][_0x398e[361]] = function(deepDataAndEvents, opt_obj2, walkers, isXML) {
      return _0x398e[708] + this[_0x398e[179]] + _0x398e[342] + this[_0x398e[709]] + _0x398e[710] + this[_0x398e[179]] + _0x398e[711] + this[_0x398e[179]] + _0x398e[366] + this[_0x398e[706]][_0x398e[361]](deepDataAndEvents, opt_obj2, walkers, isXML) + _0x398e[712] + this[_0x398e[707]][_0x398e[361]](deepDataAndEvents, opt_obj2, walkers, isXML) + _0x398e[713];
    }, val;
  }(Base);
  results[_0x398e[714]] = path;
  var resolved = function(_super) {
    /**
     * @param {?} d
     * @param {?} triggerChange
     * @param {?} value
     * @param {?} s
     * @return {undefined}
     */
    function val(d, triggerChange, value, s) {
      _super[_0x398e[431]](this, d, triggerChange, [sourceFrags[_0x398e[232]], sourceFrags[_0x398e[234]]]);
      this[_0x398e[709]] = _0x398e[715] + globals[_0x398e[176]](value) + _0x398e[491] + globals[_0x398e[176]](s) + _0x398e[716] + globals[_0x398e[172]] + _0x398e[717] + this[_0x398e[179]] + _0x398e[718] + this[_0x398e[179]] + _0x398e[719];
    }
    return __extends(val, _super), val;
  }(path);
  results[_0x398e[615]] = resolved;
  var Bridge = function(_super) {
    /**
     * @param {?} value
     * @param {?} triggerChange
     * @param {number} d
     * @return {undefined}
     */
    function val(value, triggerChange, d) {
      if (void 0 === d) {
        /** @type {Array} */
        d = [];
      }
      _super[_0x398e[431]](this);
      this[_0x398e[62]] = value;
      this[_0x398e[720]] = triggerChange;
      /** @type {Array} */
      this[_0x398e[322]] = [value];
      /** @type {number} */
      this[_0x398e[370]] = d;
    }
    return __extends(val, _super), val[_0x398e[184]][_0x398e[361]] = function(deepDataAndEvents, opt_obj2, walkers, isXML) {
      return _0x398e[31] + this[_0x398e[720]](deepDataAndEvents, opt_obj2, walkers, isXML) + this[_0x398e[62]][_0x398e[361]](deepDataAndEvents, opt_obj2, walkers, isXML);
    }, val;
  }(Base);
  results[_0x398e[721]] = Bridge;
  var modelOrCollection = function(_super) {
    /**
     * @param {?} value
     * @param {?} triggerChange
     * @param {string} d
     * @return {undefined}
     */
    function val(value, triggerChange, d) {
      _super[_0x398e[431]](this, value, function(dataAndEvents, deepDataAndEvents, ignoreMethodDoesntExist, textAlt) {
        return this[_0x398e[470]] ? _0x398e[722] + this[_0x398e[723]] + _0x398e[389] : _0x398e[722] + globals[_0x398e[178]](this[_0x398e[724]]) + _0x398e[389];
      });
      /** @type {string} */
      this[_0x398e[470]] = d;
      this[_0x398e[724]] = triggerChange;
      this[_0x398e[723]] = _0x398e[724] + this[_0x398e[179]];
      /** @type {Array} */
      this[_0x398e[374]] = d ? [this[_0x398e[723]]] : [];
    }
    return __extends(val, _super), val[_0x398e[184]][_0x398e[148]] = function(dataAndEvents) {
      if (this[_0x398e[470]]) {
        gl[_0x398e[728]](dataAndEvents[_0x398e[269]][this[_0x398e[723]]], this[_0x398e[724]][_0x398e[725]], this[_0x398e[724]][_0x398e[726]], this[_0x398e[724]][_0x398e[727]]);
      }
    }, val[_0x398e[184]][_0x398e[324]] = function() {
      return this[_0x398e[470]] ? _0x398e[481] + this[_0x398e[723]] + _0x398e[389] : _0x398e[532] + this[_0x398e[723]] + _0x398e[533] + globals[_0x398e[178]](this[_0x398e[724]]) + _0x398e[389];
    }, val;
  }(Bridge);
  results[_0x398e[610]] = modelOrCollection;
  var selectedDoc = function(_super) {
    /**
     * @param {?} value
     * @param {?} triggerChange
     * @param {string} d
     * @param {?} s
     * @return {undefined}
     */
    function val(value, triggerChange, d, s) {
      _super[_0x398e[431]](this, value, function(dataAndEvents, deepDataAndEvents, ignoreMethodDoesntExist, textAlt) {
        return _0x398e[729] + this[_0x398e[730]] + _0x398e[704] + this[_0x398e[730]] + _0x398e[731];
      });
      this[_0x398e[524]] = s;
      /** @type {string} */
      this[_0x398e[470]] = d;
      this[_0x398e[61]] = triggerChange;
      this[_0x398e[730]] = _0x398e[61] + this[_0x398e[179]];
      /** @type {Array} */
      this[_0x398e[374]] = d ? [this[_0x398e[730]], this[_0x398e[730]] + _0x398e[524], this[_0x398e[730]] + _0x398e[606]] : [];
    }
    return __extends(val, _super), val[_0x398e[184]][_0x398e[59]] = function(dataAndEvents) {
      this[_0x398e[524]] = dataAndEvents;
    }, val[_0x398e[184]][_0x398e[148]] = function(dataAndEvents) {
      if (this[_0x398e[470]]) {
        gl[_0x398e[480]](dataAndEvents[_0x398e[269]][this[_0x398e[730]]], this[_0x398e[61]]);
        gl[_0x398e[480]](dataAndEvents[_0x398e[269]][this[_0x398e[730]] + _0x398e[524]], this[_0x398e[524]]);
      }
    }, val[_0x398e[184]][_0x398e[324]] = function() {
      return this[_0x398e[470]] ? _0x398e[732] + this[_0x398e[730]] + _0x398e[733] + this[_0x398e[730]] + _0x398e[731] : _0x398e[734] + this[_0x398e[730]] + _0x398e[533] + globals[_0x398e[178]](this[_0x398e[61]]) + _0x398e[735] + this[_0x398e[730]] + _0x398e[736] + globals[_0x398e[178]](this[_0x398e[524]]) + _0x398e[389];
    }, val;
  }(Bridge);
  results[_0x398e[608]] = selectedDoc;
  var outerValidationResults = function(_super) {
    /**
     * @param {?} dataAndEvents
     * @return {undefined}
     */
    function destElements(dataAndEvents) {
      _super[_0x398e[431]](this);
      /** @type {number} */
      this[_0x398e[320]] = Base[_0x398e[643]]++;
      this[_0x398e[694]] = dataAndEvents ? _0x398e[224] : _0x398e[222];
      /** @type {Array} */
      this[_0x398e[370]] = [sourceFrags[_0x398e[230]], sourceFrags[_0x398e[224]], sourceFrags[_0x398e[222]], sourceFrags[_0x398e[220]]];
    }
    return __extends(destElements, _super), destElements[_0x398e[184]][_0x398e[361]] = function(dataAndEvents, deepDataAndEvents, ignoreMethodDoesntExist, textAlt) {
      return _0x398e[645] + this[_0x398e[320]] + _0x398e[389];
    }, destElements[_0x398e[184]][_0x398e[367]] = function() {
      return _0x398e[737] + this[_0x398e[694]] + _0x398e[738] + globals[_0x398e[171]] + _0x398e[648];
    }, destElements;
  }(Base);
  results[_0x398e[609]] = outerValidationResults;
}(mat || (mat = {}));
var kKeys = {
  Left : 37,
  Up : 38,
  Right : 39,
  Down : 40,
  Space : 32,
  Zero : 48,
  One : 49,
  Two : 50,
  Three : 51,
  Four : 52,
  Five : 53,
  Six : 54,
  Seven : 55,
  Eight : 56,
  Nine : 57,
  A : 65,
  D : 68,
  E : 69,
  F : 70,
  G : 71,
  I : 73,
  J : 74,
  K : 75,
  L : 76,
  Q : 81,
  R : 82,
  S : 83,
  W : 87,
  LastKeyCode : 222
};
var Input = function() {
  /**
   * @return {undefined}
   */
  function colorMap() {
    /** @type {Array} */
    this[_0x398e[739]] = new Array(kKeys.LastKeyCode);
    /** @type {Array} */
    this[_0x398e[740]] = new Array(kKeys.LastKeyCode);
    /** @type {Array} */
    this[_0x398e[741]] = new Array(kKeys.LastKeyCode);
    !colorMap[_0x398e[742]];
    colorMap[_0x398e[742]] = this;
  }
  return colorMap[_0x398e[151]] = function() {
    return colorMap[_0x398e[742]];
  }, colorMap[_0x398e[184]][_0x398e[743]] = function() {
    var unlock;
    /** @type {number} */
    unlock = 0;
    for (;unlock < kKeys[_0x398e[744]];unlock++) {
      /** @type {boolean} */
      this[_0x398e[740]][unlock] = false;
      /** @type {boolean} */
      this[_0x398e[739]][unlock] = false;
      /** @type {boolean} */
      this[_0x398e[741]][unlock] = false;
    }
    window[_0x398e[47]](_0x398e[745], this._onKeyUp);
    window[_0x398e[47]](_0x398e[134], this._onKeyDown);
    window[_0x398e[47]](_0x398e[746], this._onMouseDown);
    window[_0x398e[47]](_0x398e[747], this._onMouseUp);
  }, colorMap[_0x398e[184]][_0x398e[748]] = function(dataAndEvents) {
    /** @type {boolean} */
    colorMap[_0x398e[742]][_0x398e[740]][dataAndEvents[_0x398e[135]]] = false;
  }, colorMap[_0x398e[184]][_0x398e[749]] = function(dataAndEvents) {
    /** @type {boolean} */
    colorMap[_0x398e[742]][_0x398e[740]][dataAndEvents[_0x398e[135]]] = true;
  }, colorMap[_0x398e[184]][_0x398e[750]] = function(dataAndEvents) {
  }, colorMap[_0x398e[184]][_0x398e[751]] = function(dataAndEvents) {
  }, colorMap[_0x398e[184]][_0x398e[148]] = function() {
    var currentParam;
    /** @type {number} */
    currentParam = 0;
    for (;currentParam < kKeys[_0x398e[744]];currentParam++) {
      this[_0x398e[741]][currentParam] = !this[_0x398e[739]][currentParam] && this[_0x398e[740]][currentParam];
      this[_0x398e[739]][currentParam] = this[_0x398e[740]][currentParam];
    }
  }, colorMap[_0x398e[184]][_0x398e[752]] = function(timeoutKey) {
    return this[_0x398e[740]][timeoutKey];
  }, colorMap[_0x398e[184]][_0x398e[753]] = function(timeoutKey) {
    return this[_0x398e[741]][timeoutKey];
  }, colorMap[_0x398e[742]] = new colorMap, colorMap;
}();
var HDRLoader;
!function(dataAndEvents) {
  /**
   * @param {Object} _gl
   * @param {(Array|Float32Array)} unit
   * @param {number} obj
   * @param {number} value
   * @param {?} item
   * @return {?}
   */
  function iterator(_gl, unit, obj, value, item) {
    var r20 = _gl[_0x398e[314]]();
    return _gl[_0x398e[316]](_gl.TEXTURE_2D, r20), _gl[_0x398e[318]](_gl.TEXTURE_2D, _gl.TEXTURE_MIN_FILTER, _gl.LINEAR), _gl[_0x398e[318]](_gl.TEXTURE_2D, _gl.TEXTURE_MAG_FILTER, _gl.LINEAR), _gl[_0x398e[318]](_gl.TEXTURE_2D, _gl.TEXTURE_WRAP_S, _gl.REPEAT), _gl[_0x398e[318]](_gl.TEXTURE_2D, _gl.TEXTURE_WRAP_T, _gl.CLAMP_TO_EDGE), _gl[_0x398e[754]](_gl.UNPACK_FLIP_Y_WEBGL, false), _gl[_0x398e[317]](_gl.TEXTURE_2D, 0, _gl.RGB16F, obj, value, 0, item, _gl.FLOAT, unit), _gl[_0x398e[316]](_gl.TEXTURE_2D, 
    null), r20;
  }
  /**
   * @param {Object} value
   * @param {?} opt_obj2
   * @param {number} a
   * @param {number} obj
   * @param {?} relativeToItem
   * @param {?} $i18next
   * @return {undefined}
   */
  function insertBefore(value, opt_obj2, a, obj, relativeToItem, $i18next) {
    /** @type {XMLHttpRequest} */
    var req = new XMLHttpRequest;
    req[_0x398e[293]](_0x398e[292], opt_obj2, true);
    req[_0x398e[755]] = _0x398e[756];
    /**
     * @param {?} dataAndEvents
     * @return {undefined}
     */
    req[_0x398e[315]] = function(dataAndEvents) {
      var array = req[_0x398e[757]];
      if (array) {
        /** @type {Uint8Array} */
        var view = new Uint8Array(array);
        /** @type {Float32Array} */
        var y = new Float32Array(a * obj * 3);
        /** @type {number} */
        var i = 0;
        for (;i < view[_0x398e[3]];i++) {
          if (10 === view[i] && 10 === view[i + 1]) {
            i += 2;
            break;
          }
        }
        for (;i < view[_0x398e[3]];i++) {
          if (10 === view[i]) {
            i += 1;
            break;
          }
        }
        /** @type {number} */
        var j = 0;
        /** @type {number} */
        var max = 0;
        for (;obj > max;max++) {
          /** @type {number} */
          var b = 0;
          for (;a > b;b++) {
            var table = view[i++];
            var c = view[i++];
            var style = view[i++];
            var val = view[i++];
            var t = Math[_0x398e[758]](2, val - 128);
            /** @type {number} */
            y[j++] = table / 256 * t;
            /** @type {number} */
            y[j++] = c / 256 * t;
            /** @type {number} */
            y[j++] = style / 256 * t;
          }
        }
        var key = iterator(value, y, a, obj, relativeToItem);
        $i18next(key);
      } else {
        $i18next(null);
      }
    };
    req[_0x398e[294]](null);
  }
  /** @type {function (Object, (Array|Float32Array), number, number, ?): ?} */
  dataAndEvents[_0x398e[759]] = iterator;
  /** @type {function (Object, ?, number, number, ?, ?): undefined} */
  dataAndEvents[_0x398e[44]] = insertBefore;
}(HDRLoader || (HDRLoader = {}));
var toneMap;
!function(dataAndEvents) {
  dataAndEvents[_0x398e[86]] = new ShaderProgram;
  dataAndEvents[_0x398e[86]][_0x398e[278]](globals[_0x398e[174]], gl.VERTEX_SHADER, mode[_0x398e[268]]);
  dataAndEvents[_0x398e[86]][_0x398e[278]](_0x398e[760], gl.FRAGMENT_SHADER, mode[_0x398e[268]]);
  dataAndEvents[_0x398e[86]][_0x398e[283]]();
  dataAndEvents[_0x398e[86]][_0x398e[272]]([_0x398e[413]]);
  dataAndEvents[_0x398e[86]][_0x398e[275]]([_0x398e[761]]);
  dataAndEvents[_0x398e[90]] = new ShaderProgram;
  dataAndEvents[_0x398e[90]][_0x398e[278]](globals[_0x398e[174]], gl.VERTEX_SHADER, mode[_0x398e[268]]);
  dataAndEvents[_0x398e[90]][_0x398e[278]](_0x398e[762], gl.FRAGMENT_SHADER, mode[_0x398e[268]]);
  dataAndEvents[_0x398e[90]][_0x398e[283]]();
  dataAndEvents[_0x398e[90]][_0x398e[272]]([_0x398e[413]]);
  dataAndEvents[_0x398e[90]][_0x398e[275]]([_0x398e[761], _0x398e[51]]);
  dataAndEvents[_0x398e[92]] = new ShaderProgram;
  dataAndEvents[_0x398e[92]][_0x398e[278]](globals[_0x398e[174]], gl.VERTEX_SHADER, mode[_0x398e[268]]);
  dataAndEvents[_0x398e[92]][_0x398e[278]](_0x398e[763], gl.FRAGMENT_SHADER, mode[_0x398e[268]]);
  dataAndEvents[_0x398e[92]][_0x398e[283]]();
  dataAndEvents[_0x398e[92]][_0x398e[272]]([_0x398e[413]]);
  dataAndEvents[_0x398e[92]][_0x398e[275]]([_0x398e[761], _0x398e[51]]);
  dataAndEvents[_0x398e[94]] = new ShaderProgram;
  dataAndEvents[_0x398e[94]][_0x398e[278]](globals[_0x398e[174]], gl.VERTEX_SHADER, mode[_0x398e[268]]);
  dataAndEvents[_0x398e[94]][_0x398e[278]](_0x398e[764], gl.FRAGMENT_SHADER, mode[_0x398e[268]]);
  dataAndEvents[_0x398e[94]][_0x398e[283]]();
  dataAndEvents[_0x398e[94]][_0x398e[272]]([_0x398e[413]]);
  dataAndEvents[_0x398e[94]][_0x398e[275]]([_0x398e[761], _0x398e[51]]);
  dataAndEvents[_0x398e[96]] = new ShaderProgram;
  dataAndEvents[_0x398e[96]][_0x398e[278]](globals[_0x398e[174]], gl.VERTEX_SHADER, mode[_0x398e[268]]);
  dataAndEvents[_0x398e[96]][_0x398e[278]](_0x398e[765], gl.FRAGMENT_SHADER, mode[_0x398e[268]]);
  dataAndEvents[_0x398e[96]][_0x398e[283]]();
  dataAndEvents[_0x398e[96]][_0x398e[272]]([_0x398e[413]]);
  dataAndEvents[_0x398e[96]][_0x398e[275]]([_0x398e[761], _0x398e[51]]);
  dataAndEvents[_0x398e[98]] = new ShaderProgram;
  dataAndEvents[_0x398e[98]][_0x398e[278]](globals[_0x398e[174]], gl.VERTEX_SHADER, mode[_0x398e[268]]);
  dataAndEvents[_0x398e[98]][_0x398e[278]](_0x398e[766], gl.FRAGMENT_SHADER, mode[_0x398e[268]]);
  dataAndEvents[_0x398e[98]][_0x398e[283]]();
  dataAndEvents[_0x398e[98]][_0x398e[272]]([_0x398e[413]]);
  dataAndEvents[_0x398e[98]][_0x398e[275]]([_0x398e[761], _0x398e[51]]);
}(toneMap || (toneMap = {}));
var RenderMode = function() {
  /**
   * @param {?} dataAndEvents
   * @param {?} deepDataAndEvents
   * @return {undefined}
   */
  function clone(dataAndEvents, deepDataAndEvents) {
    /** @type {number} */
    this[_0x398e[81]] = 32;
    this[_0x398e[767]] = dataAndEvents;
    this[_0x398e[408]] = deepDataAndEvents;
    this[_0x398e[768]] = toneMap[_0x398e[90]];
    this[_0x398e[769]] = Date[_0x398e[146]]();
    this[_0x398e[743]]();
  }
  return clone[_0x398e[184]][_0x398e[37]] = function(deepDataAndEvents, opt_obj2, walkers, isXML) {
    if (void 0 === isXML) {
      /** @type {number} */
      isXML = 1;
    }
    if (this[_0x398e[40]]) {
      this[_0x398e[40]][_0x398e[311]]();
    }
    this[_0x398e[770]](deepDataAndEvents, opt_obj2, walkers, isXML);
  }, clone[_0x398e[184]][_0x398e[771]] = function() {
    var FORMAT = gl[_0x398e[772]];
    var r20 = gl[_0x398e[314]]();
    return gl[_0x398e[316]](gl.TEXTURE_2D, r20), gl[_0x398e[318]](gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR), gl[_0x398e[318]](gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR), gl[_0x398e[318]](gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE), gl[_0x398e[318]](gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE), gl[_0x398e[317]](gl.TEXTURE_2D, 0, gl.RGB, this[_0x398e[408]][0], this[_0x398e[408]][1], 0, gl.RGB, FORMAT, null), gl[_0x398e[316]](gl.TEXTURE_2D, null), r20;
  }, clone[_0x398e[184]][_0x398e[743]] = function() {
    this[_0x398e[773]] = gl[_0x398e[554]]();
    gl[_0x398e[555]](gl.ARRAY_BUFFER, this[_0x398e[773]]);
    gl[_0x398e[556]](gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
    this[_0x398e[774]] = gl[_0x398e[775]]();
    /** @type {Array} */
    this[_0x398e[776]] = new Array(2);
    this[_0x398e[776]][0] = this[_0x398e[771]]();
    this[_0x398e[776]][1] = this[_0x398e[771]]();
    /** @type {number} */
    this[_0x398e[777]] = 0;
    /** @type {number} */
    this[_0x398e[51]] = 1;
  }, clone[_0x398e[184]][_0x398e[311]] = function() {
    gl[_0x398e[778]](this[_0x398e[773]]);
    gl[_0x398e[779]](this[_0x398e[774]]);
    gl[_0x398e[42]](this[_0x398e[776]][0]);
    gl[_0x398e[42]](this[_0x398e[776]][1]);
  }, clone[_0x398e[184]][_0x398e[41]] = function() {
    /** @type {number} */
    this[_0x398e[777]] = 0;
  }, clone[_0x398e[184]][_0x398e[149]] = function() {
    if (this[_0x398e[40]] && !(this[_0x398e[777]] > this[_0x398e[81]] && this[_0x398e[81]] > 0)) {
      this[_0x398e[40]][_0x398e[309]]();
      this[_0x398e[767]](this[_0x398e[40]]);
      /**
       * @return {?}
       */
      var throttledUpdate = function() {
        return this[_0x398e[780]] = 18030 * (65535 & this[_0x398e[780]]) + (this[_0x398e[780]] << 16), this[_0x398e[781]] = 30903 * (65535 & this[_0x398e[781]]) + (this[_0x398e[781]] << 16), this[_0x398e[780]] << 16 + (65535 & this[_0x398e[781]]);
      };
      gl[_0x398e[417]](this[_0x398e[40]][_0x398e[269]][_0x398e[405]], Date[_0x398e[146]]() - this[_0x398e[769]] + throttledUpdate());
      gl[_0x398e[417]](this[_0x398e[40]][_0x398e[269]][_0x398e[406]], this[_0x398e[777]] / (this[_0x398e[777]] + 1));
      gl[_0x398e[782]](this[_0x398e[40]][_0x398e[269]][_0x398e[408]], this[_0x398e[408]]);
      gl[_0x398e[436]](gl.TEXTURE0);
      gl[_0x398e[316]](gl.TEXTURE_2D, this[_0x398e[776]][1]);
      gl[_0x398e[437]](this[_0x398e[40]][_0x398e[269]][_0x398e[407]], 0);
      gl[_0x398e[555]](gl.ARRAY_BUFFER, this[_0x398e[773]]);
      gl[_0x398e[783]](gl.FRAMEBUFFER, this[_0x398e[774]]);
      gl[_0x398e[784]](gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this[_0x398e[776]][0], 0);
      gl[_0x398e[785]](this[_0x398e[40]][_0x398e[270]][_0x398e[413]]);
      gl[_0x398e[786]](this[_0x398e[40]][_0x398e[270]][_0x398e[413]], 2, gl.FLOAT, false, 0, 0);
      gl[_0x398e[787]](gl.TRIANGLE_STRIP, 0, 4);
      gl[_0x398e[783]](gl.FRAMEBUFFER, null);
      gl[_0x398e[316]](gl.TEXTURE_2D, null);
      this[_0x398e[777]]++;
    }
  }, clone[_0x398e[184]][_0x398e[87]] = function(dataAndEvents) {
    this[_0x398e[768]] = dataAndEvents;
  }, clone[_0x398e[184]][_0x398e[150]] = function() {
    this[_0x398e[768]][_0x398e[309]]();
    gl[_0x398e[417]](this[_0x398e[768]][_0x398e[269]][_0x398e[51]], this[_0x398e[51]]);
    gl[_0x398e[436]](gl.TEXTURE0);
    gl[_0x398e[316]](gl.TEXTURE_2D, this[_0x398e[776]][0]);
    gl[_0x398e[555]](gl.ARRAY_BUFFER, this[_0x398e[773]]);
    gl[_0x398e[785]](this[_0x398e[768]][_0x398e[270]][_0x398e[413]]);
    gl[_0x398e[786]](this[_0x398e[768]][_0x398e[270]][_0x398e[413]], 2, gl.FLOAT, false, 0, 0);
    gl[_0x398e[787]](gl.TRIANGLE_STRIP, 0, 4);
    gl[_0x398e[316]](gl.TEXTURE_2D, null);
    this[_0x398e[776]][_0x398e[788]]();
  }, clone;
}();
__extends = this && this[_0x398e[426]] || function(elems, parent) {
  /**
   * @return {undefined}
   */
  function result() {
    /** @type {Function} */
    this[_0x398e[427]] = elems;
  }
  var i;
  for (i in parent) {
    if (parent[_0x398e[428]](i)) {
      elems[i] = parent[i];
    }
  }
  elems[_0x398e[184]] = null === parent ? Object[_0x398e[429]](parent) : (result[_0x398e[184]] = parent[_0x398e[184]], new result);
};
var RayTracer = function(_super) {
  /**
   * @param {?} value
   * @param {?} d
   * @return {undefined}
   */
  function val(value, d) {
    _super[_0x398e[431]](this, value, d);
  }
  return __extends(val, _super), val[_0x398e[184]][_0x398e[770]] = function(text, deepDataAndEvents, dataAndEvents, opt_obj2) {
    this[_0x398e[40]] = utility[_0x398e[404]](text, deepDataAndEvents, environment, opt_obj2);
  }, val;
}(RenderMode);
__extends = this && this[_0x398e[426]] || function(elems, parent) {
  /**
   * @return {undefined}
   */
  function result() {
    /** @type {Function} */
    this[_0x398e[427]] = elems;
  }
  var i;
  for (i in parent) {
    if (parent[_0x398e[428]](i)) {
      elems[i] = parent[i];
    }
  }
  elems[_0x398e[184]] = null === parent ? Object[_0x398e[429]](parent) : (result[_0x398e[184]] = parent[_0x398e[184]], new result);
};
var SphericalHarmonics = function(_super) {
  /**
   * @param {?} value
   * @param {?} d
   * @param {?} s
   * @return {undefined}
   */
  function val(value, d, s) {
    if (_super[_0x398e[431]](this, value, d), this[_0x398e[125]] = sourceFrags[s], !this[_0x398e[125]]) {
      throw _0x398e[789];
    }
  }
  return __extends(val, _super), val[_0x398e[184]][_0x398e[770]] = function(text, deepDataAndEvents, dataAndEvents, opt_obj2) {
    this[_0x398e[40]] = utility[_0x398e[403]](text, deepDataAndEvents, environment, opt_obj2, this[_0x398e[125]]);
  }, val;
}(RenderMode);
__extends = this && this[_0x398e[426]] || function(elems, parent) {
  /**
   * @return {undefined}
   */
  function result() {
    /** @type {Function} */
    this[_0x398e[427]] = elems;
  }
  var i;
  for (i in parent) {
    if (parent[_0x398e[428]](i)) {
      elems[i] = parent[i];
    }
  }
  elems[_0x398e[184]] = null === parent ? Object[_0x398e[429]](parent) : (result[_0x398e[184]] = parent[_0x398e[184]], new result);
};
var PathTracer = function(_super) {
  /**
   * @param {?} value
   * @param {?} d
   * @return {undefined}
   */
  function val(value, d) {
    _super[_0x398e[431]](this, value, d);
  }
  return __extends(val, _super), val[_0x398e[184]][_0x398e[770]] = function(text, deepDataAndEvents, dataAndEvents, opt_obj2) {
    this[_0x398e[40]] = utility[_0x398e[404]](text, deepDataAndEvents, environment, opt_obj2);
  }, val;
}(RenderMode);
var sl;
/** @type {boolean} */
var loading = true;
var lightSphere;
updateXML(_0x398e[790]);
var fps;
!function(parent) {
  /**
   * @return {undefined}
   */
  function t() {
    if (!aposed) {
      aposed = (new Date)[_0x398e[791]]();
      /** @type {number} */
      quoted = 0;
    }
    /** @type {number} */
    var lowestDeltaXY = ((new Date)[_0x398e[791]]() - aposed) / 1E3;
    if (aposed = (new Date)[_0x398e[791]](), quoted = Math[_0x398e[792]](1 / lowestDeltaXY), _0x6f51x5 >= 60) {
      var delta = chunk[_0x398e[793]](function(far, near) {
        return far + near;
      });
      var currentValue = Math[_0x398e[792]](delta / chunk[_0x398e[3]]);
      /** @type {number} */
      _0x6f51x5 = 0;
      _0x6f51x2[_0x398e[131]] = Math[_0x398e[794]](currentValue) + _0x398e[795];
    } else {
      if (quoted !== 1 / 0) {
        chunk[_0x398e[282]](quoted);
      }
      _0x6f51x5++;
    }
  }
  var aposed;
  var quoted;
  var _0x6f51x2 = document[_0x398e[24]](_0x398e[796]);
  /** @type {number} */
  var _0x6f51x5 = 0;
  /** @type {Array} */
  var chunk = [];
  /** @type {function (): undefined} */
  parent[_0x398e[148]] = t;
}(fps || (fps = {}));
/** @type {null} */
var tracerIntegrator = null;
var currentSceneObjects;
var environment;
var projection;
/** @type {number} */
var bounces = 8;
/** @type {number} */
var samples = 32;
var camera;
var environmentRotation;
var envTexture;
var startTime;
var shutterOpenTime;
var v;
/** @type {number} */
var normalizedGlossiness = 0.3;
/**
 * @param {?} deepDataAndEvents
 * @return {undefined}
 */
var integratorCallBack = function(deepDataAndEvents) {
  deepDataAndEvents[_0x398e[309]]();
  v = camera.GetViewMatrix();
  gl[_0x398e[797]](deepDataAndEvents[_0x398e[269]][_0x398e[371]], false, new Float32Array([v[10], v[9], v[8], v[6], v[5], v[4], v[2], v[1], v[0]]));
  gl[_0x398e[480]](deepDataAndEvents[_0x398e[269]][_0x398e[372]], camera[_0x398e[450]]);
  gl[_0x398e[798]](deepDataAndEvents[_0x398e[269]][_0x398e[373]], startTime, startTime + shutterOpenTime);
  /** @type {number} */
  var unlock = 0;
  for (;unlock < currentSceneObjects[_0x398e[3]];unlock++) {
    currentSceneObjects[unlock][_0x398e[148]](deepDataAndEvents);
    currentSceneObjects[unlock][_0x398e[62]][_0x398e[148]](deepDataAndEvents);
  }
  projection[_0x398e[148]](deepDataAndEvents);
  environment[_0x398e[148]](deepDataAndEvents);
};
window[_0x398e[47]](_0x398e[799], function() {
  if (gl) {
    setStatus(_0x398e[43]);
    Input[_0x398e[151]]()[_0x398e[743]]();
    initialize();
  } else {
    setStatus(_0x398e[800] + failureMessage);
  }
});
