/*
OpenCL RandomForestClassifier
classifier_class_name = ObjectClassifier
feature_specification = area mean_intensity standard_deviation_intensity mean_max_distance_to_centroid_ratio
num_ground_truth_dimensions = 1
num_classes = 3
num_features = 4
max_depth = 2
num_trees = 10
apoc_version = 0.6.1
*/
__kernel void predict (IMAGE_in0_TYPE in0, IMAGE_in1_TYPE in1, IMAGE_in2_TYPE in2, IMAGE_in3_TYPE in3, IMAGE_out_TYPE out) {
 sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
 const int x = get_global_id(0);
 const int y = get_global_id(1);
 const int z = get_global_id(2);
 float i0 = READ_IMAGE(in0, sampler, POS_in0_INSTANCE(x,y,z,0)).x;
 float i1 = READ_IMAGE(in1, sampler, POS_in1_INSTANCE(x,y,z,0)).x;
 float i2 = READ_IMAGE(in2, sampler, POS_in2_INSTANCE(x,y,z,0)).x;
 float i3 = READ_IMAGE(in3, sampler, POS_in3_INSTANCE(x,y,z,0)).x;
 float s0=0;
 float s1=0;
 float s2=0;
if(i3<1.859057903289795){
 s0+=5.0;
} else {
 if(i2<15.608141899108887){
  s2+=1.0;
 } else {
  s1+=1.0;
 }
}
if(i3<1.7305941581726074){
 s0+=3.0;
} else {
 if(i3<2.0542917251586914){
  s1+=2.0;
 } else {
  s2+=2.0;
 }
}
if(i0<19723.0){
 if(i3<1.9892064332962036){
  s0+=3.0;
 } else {
  s2+=1.0;
 }
} else {
 s1+=3.0;
}
if(i0<5851.5){
 s2+=2.0;
} else {
 if(i0<15677.5){
  s0+=3.0;
 } else {
  s1+=2.0;
 }
}
if(i3<1.99391770362854){
 if(i3<1.7305941581726074){
  s0+=2.0;
 } else {
  s1+=1.0;
 }
} else {
 s2+=4.0;
}
if(i3<1.9892064332962036){
 s0+=6.0;
} else {
 s2+=1.0;
}
if(i3<1.7986836433410645){
 s0+=4.0;
} else {
 s1+=3.0;
}
if(i0<15080.0){
 s0+=4.0;
} else {
 s1+=3.0;
}
if(i3<1.7909682989120483){
 s0+=5.0;
} else {
 if(i0<27238.0){
  s2+=1.0;
 } else {
  s1+=1.0;
 }
}
if(i0<4240.5){
 s2+=2.0;
} else {
 s0+=5.0;
}
 float max_s=s0;
 int cls=1;
 if (max_s < s1) {
  max_s = s1;
  cls=2;
 }
 if (max_s < s2) {
  max_s = s2;
  cls=3;
 }
 WRITE_IMAGE (out, POS_out_INSTANCE(x,y,z,0), cls);
}
