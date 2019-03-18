

'''
#you need to COPY the code below in your CNN code, after the initialization part and before the training part
#you should also give THREE parameters to the script: one IMAGE(or the IMAGES you want to test) from your test images; OUTPUT of the last layer; your SESSION(sess)
#ATTENTION!! This script well functions in the code with the layers seperated. If you use the 'DEF' to write the layers, you have to give the NAMES to the layers.

#################################
import script
image_for_structure={x: data.test.images[:1]}
script.func(sess,output,image_for_structure)
#################################
'''

import tensorflow as tf
import sys
import prettytable as pt
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.profiler import option_builder
from tensorflow.python.profiler import model_analyzer
import linecache
import math


# Print to stdout an analysis of the memory usage and the timing information
# broken down by operation types.
def func(sess,output,test0): # this programme needs to run this model one time first, but except the training part, so it will be quick
 run_metadata=tf.RunMetadata()
 sess.run(output,
          test0,
          options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True),
          run_metadata=run_metadata)

 memory_stats=tf.profiler.profile( # this tensorflow fonction is to profile the memory consumption by using graph node name scope
     tf.get_default_graph(),
     run_meta=run_metadata,
     cmd='scope',
     options=tf.profiler.ProfileOptionBuilder.time_and_memory())
 
########################################################################################################__________________________________________________
 sum_memory = 0
 sum_memory1 = 0
 tb = pt.PrettyTable()

 tb.field_names = ["Layer Name","Shapes(size of matrix for caculation)"," Input Requested Bytes & Explanation "," Output Requested Bytes & Explanation "] 

 # using prettytable to collect information, howere prettytable has one big probleme, that is each colume has limit, so we can't write too many things in one colume, and there're not many solutions in source code so I built about 3 rows for each layer so that we can print the comments as well. But if you want to add someting, you need to add another row first, otherwise the table will not be "pretty"
 a0_for_layername=0 # helping extract the simple node name
 a1=0 # from a1 to a3, they are valeurs to solve the "diffrent name problem". "Diffrent name problem" is the training variables'names are diffrent from the operation steps'names. In training variables, from exemple names are"conv2d,conv2d_1,conv2d_2" but in operation step's names they are "conv2d,conv2d_2,con2d_3" if they have the names which are given from the programmer, the names will change again, to "conv2d,xxx,conv2d_1..." or "xxx,conv2d,conv2d_1".
 a2=0  
 a3=0 
 a4=0 # when we find out the operations correspond to the layer name the one we get from the training variables in order, a4 is to find the last operation correspond to this layer, to find out the activation fonction and the pooling layers because they are not in the training variables
 flag1=0 # from flag1 to flag3, they are the flags when we have already get the results we want in one term, we can end the iteration directly in case there are other operations who has the traing variables'names but don't have the actual node in graph 
 flag2=0
 flag3=0
 flag=0 # flag for the first time running, to get the information of the image input
 count_dense=0#to count the num of FC layers for printing correct comments
 count = len(tf.trainable_variables())/2 # to count the num of trainable variables
 count_conv = 0 #to help count_dense to know how many FC layers rest
# we first use the training varibales to get the convlution layers and the dense layers, then we can get their operations steps, after that we can get the activation fonction and the pooling layers, we can also add the other layers checking after them 
 max_mem=[0,0,0,0,0,0]
 min_mem=[0,0,0,0,0,0]
 list_mem=[]
 list_mem1=[]
 list_mem2=[]
 list_mem3=[]
 list_mem0=[]
 list_mem32=[]
 list_mem16=[]
 list_mem8=[]
 list_mem6=[]
 list_mem4=[]
 list_mem=[list_mem0,list_mem32,list_mem16,list_mem8,list_mem6,list_mem4]
 for v in tf.trainable_variables():
   if(("/kernel:0" in v.name) or ("/Variable:0" in v.name)): # to get the layer name
    a0_for_layername=v.name.index("/") #before the "/" is the name of this layer, a0 is to get the index of "/"
    v1=v.name[0:a0_for_layername] #to get the name of the convolution layer, because some programmer like to rename the layers
    v2=v1+"/" # v2 is for the juge in next steps
    a0_for_layername=0
    a3=a3+1
    if((a3-a1)<=1):
     if(v1=="conv2d"):
      a2=a2+1
     else:
      a1=a1+1
    if((a3-a1)>1):
     if(v1==("conv2d_"+str(a2))):
      a2=a2+1
      v1="conv2d_"+str(a2)
     else:
      a1=a1+1
    if((len(v.get_shape().as_list())==4)): # to juge the convolution layers
      count_conv = count_conv+1
      val = v.get_shape().as_list() # preserve the shape to simplify the code, it can show the shape of the layer

      for a in memory_stats.children:
       with open("memory_stats.txt", "w") as out:
        out.write(str(a))
       with open("memory_stats.txt", "r") as fr:
        for line in fr:
         if (("name: "+"\""+v1+"\"") in line):
           linecache.updatecache('memory_stats.txt')
           if(flag==0): #print image line
             tb.add_row(["       ",("["+str(linecache.getline('memory_stats.txt',16).split(' ')[9].strip())+","+str(linecache.getline('memory_stats.txt',19).split(' ')[9].strip())+","+str(linecache.getline('memory_stats.txt',22).split(' ')[9].strip())+","+str(linecache.getline('memory_stats.txt',25).split(' ')[9].strip())+"]"),"                                       ","                                 "])
             tb.add_row(["(input image)","The shape and number of test image(s),\nwe have "+str(linecache.getline('memory_stats.txt',25).split(' ')[9].strip())+" image(s)"+"("+str(linecache.getline('memory_stats.txt',16).split(' ')[9].strip())+","+str(linecache.getline('memory_stats.txt',19).split(' ')[9].strip())+","+str(linecache.getline('memory_stats.txt',22).split(' ')[9].strip())+")","                                      ","                                 "])
             
             tb.add_row(["----------------------------","------------------------------------------------------------","---------------------------------------------------------","------------------------------------------"])
             sum_memory = sum_memory + (int(linecache.getline('memory_stats.txt',16).split(' ')[9].strip()))*(int(linecache.getline('memory_stats.txt',19).split(' ')[9].strip()))*(int(linecache.getline('memory_stats.txt',22).split(' ')[9].strip()))*(int(linecache.getline('memory_stats.txt',25).split(' ')[9].strip()))*4
             sum_memory1 = sum_memory1 + (int(linecache.getline('memory_stats.txt',16).split(' ')[9].strip()))*(int(linecache.getline('memory_stats.txt',19).split(' ')[9].strip()))*(int(linecache.getline('memory_stats.txt',22).split(' ')[9].strip()))*(int(linecache.getline('memory_stats.txt',25).split(' ')[9].strip()))*4
             flag=1
             
           input_bytes_convolution= (int(linecache.getline('memory_stats.txt',16).split(' ')[9].strip()))*(int(linecache.getline('memory_stats.txt',19).split(' ')[9].strip()))*(int(linecache.getline('memory_stats.txt',22).split(' ')[9].strip()))*(int(linecache.getline('memory_stats.txt',25).split(' ')[9].strip()))*4+val[0]*val[1]*val[2]*val[3]*4
           list_mem0.append(input_bytes_convolution)
           output_bytes_convolution = int(linecache.getline('memory_stats.txt',3).split(' ')[1].strip()) #output_bytes means the memory consumpotion of the data we can get after this layer.

           
           min_mem[0]=input_bytes_convolution
           if(input_bytes_convolution>max_mem[0]):
            max_mem[0]=input_bytes_convolution
           if(input_bytes_convolution<min_mem[0]):
            min_mem[0] = input_bytes_convolution

            
           flag1=1  
           output_size_convolution = math.sqrt(output_bytes_convolution/4/val[3]) #print convolution layers
           output_size_convolution = int(output_size_convolution)
           sum_memory = val[0]*val[1]*val[2]*val[3]*4+output_bytes_convolution+sum_memory
           sum_memory1 = sum_memory1+output_bytes_convolution+val[0]*val[1]*val[2]*val[3]*4
           
           input_convolution32 = input_bytes_convolution*8/1000
           list_mem32.append(input_convolution32)
           input_convolution16 = input_bytes_convolution*4/1000
           list_mem16.append(input_convolution16)
           input_convolution8 = input_bytes_convolution*2/1000
           list_mem8.append(input_convolution8)
           input_convolution6 = input_bytes_convolution*1.5/1000
           list_mem6.append(input_convolution6)
           input_convolution4 = input_bytes_convolution/1000
           list_mem4.append(input_convolution4)
           

           min_mem[1]=input_convolution32
           if(input_convolution32>max_mem[1]):
            max_mem[1]=input_convolution32
           if(input_convolution32<min_mem[1]):
            min_mem[1] = input_convolution32
                       
           min_mem[2]=input_convolution16
           if(input_convolution16>max_mem[2]):
            max_mem[2]=input_convolution16
           if(input_convolution16<min_mem[2]):
            min_mem[2] = input_convolution16
                       
           min_mem[3]=input_convolution8
           if(input_convolution8>max_mem[3]):
            max_mem[3]=input_convolution8
           if(input_convolution8<min_mem[3]):
            min_mem[3] = input_convolution8
            
           min_mem[4]=input_convolution6
           if(input_convolution6>max_mem[4]):
            max_mem[4]=input_convolution6
           if(input_convolution6<min_mem[4]):
            min_mem[4] = input_convolution6
           
           
           min_mem[5]=input_convolution4
           if(input_convolution4>max_mem[5]):
            max_mem[5]=input_convolution4
           if(input_convolution4<min_mem[5]):
            min_mem[5] = input_convolution4
           
           tb.add_row(["       ",val,str(input_bytes_convolution/1000)+"KB",str(output_bytes_convolution/1000)+"KB"]) 
           tb.add_row([v2[:-1],"In this convolution layer, we have "+str(val[3])+" filters"+"("+str(val[0])+"x"+str(val[1])+"x"+str(val[2])+")"+"\nso here, we have actually "+str(val[3])+" matrices after this layer.","Input filters of this layer + Matrices from last layer =\n"+"("+str(val[0])+"x"+str(val[1])+"x"+str(val[2])+"x"+str(val[3])+" + "+str(linecache.getline('memory_stats.txt',16).split(' ')[9].strip())+"x"+str(linecache.getline('memory_stats.txt',19).split(' ')[9].strip())+"x"+str(linecache.getline('memory_stats.txt',22).split(' ')[9].strip())+"x"+str(linecache.getline('memory_stats.txt',25).split(' ')[9].strip())+")"+"x"+"4"+" = "+str(input_bytes_convolution),"There're "+str(val[3])+" output matrices"+"("+str(linecache.getline('memory_stats.txt',16).split(' ')[9].strip())+","+str(output_size_convolution)+","+str(output_size_convolution)+")"+"\n"+"each matrice corresponding one filter."])
           tb.add_row([" "," "," "," "])
           tb.add_row([" "," ","size of data:"+"32bits  "+str(input_bytes_convolution*8/1000)+"Kbit","size of data:"+"32bits  "+str(output_bytes_convolution*8/1000)+"Kbit"])
           tb.add_row([" "," ","             "+"16bits  "+str(input_bytes_convolution*4/1000)+"Kbit","size of data:"+"16bits  "+str(output_bytes_convolution*4/1000)+"Kbit"])
           tb.add_row([" "," ","             "+"8bits  "+str(input_bytes_convolution*2/1000)+"Kbit","size of data:"+"8bits  "+str(output_bytes_convolution*2/1000)+"Kbit"])
           tb.add_row([" "," ","             "+"6bits  "+str(input_bytes_convolution*1.5/1000)+"Kbit","size of data:"+"6bits  "+str(output_bytes_convolution*1.5/1000)+"Kbit"])
           tb.add_row([" "," ","             "+"4bits  "+str(input_bytes_convolution/1000)+"Kbit","size of data:"+"4bits  "+str(output_bytes_convolution/1000)+"Kbit"])
           tb.add_row(["----------------------------","------------------------------------------------------------","---------------------------------------------------------","------------------------------------------"])
           for v3 in tf.get_default_graph().get_operations(): # to find out the activation fonctions and the pooling layers. 
            a4=a4+1
            if(v1 in v3.name):
             break
           while(v1 in tf.get_default_graph().get_operations()[a4].name):# We can get the operation names from the layer name exists to the name disappear for the first time. The last operation of one convolution layer is BiasAdd if there is no activation fonction, so we can stop at there and start to find out the other layers
             a4=a4+1
             if(("/BiasAdd" in tf.get_default_graph().get_operations()[a4].name) or ("/add" in tf.get_default_graph().get_operations()[a4].name)):
              break
           
           if((v1 in tf.get_default_graph().get_operations()[a4+1].name) or ("relu" in tf.get_default_graph().get_operations()[a4+1].name) or ("dropout" in tf.get_default_graph().get_operations()[a4+1].name) or ("sigmoid" in tf.get_default_graph().get_operations()[a4+1].name) or("tanh" in tf.get_default_graph().get_operations()[a4+1].name)):
              activation_layer_nameindex = tf.get_default_graph().get_operations()[a4+1].name.index("/")
              activation_layer_name = tf.get_default_graph().get_operations()[a4+1].name[(activation_layer_nameindex+1):]
              activation_layer= "activation layer"#the name of activation_layer 
              sum_memory = sum_memory + output_bytes_convolution
              
              list_mem0.append(output_bytes_convolution)   
                         
              if(output_bytes_convolution>max_mem[0]):
               max_mem[0]=output_bytes_convolution
              if(output_bytes_convolution<min_mem[0]):
               min_mem[0] = output_bytes_convolution
              
              
              input_activation32=output_bytes_convolution*8/1000
              list_mem32.append(input_activation32)  
              input_activation16=output_bytes_convolution*4/1000
              list_mem16.append(input_activation16)  
              input_activation8=output_bytes_convolution*2/1000
              list_mem8.append(input_activation8)  
              input_activation6=output_bytes_convolution*1.5/1000
              list_mem6.append(input_activation6)  
              input_activation4=output_bytes_convolution*1/1000
              list_mem4.append(input_activation4)  
              
              if(input_activation32>max_mem[1]):
               max_mem[1]=input_activation32
              if(input_activation32<min_mem[1]):
               min_mem[1] = input_activation32
               
              if(input_activation16>max_mem[2]):
               max_mem[2]=input_activation16
              if(input_activation16<min_mem[2]):
               min_mem[2] = input_activation16
               
              if(input_activation8>max_mem[3]):
               max_mem[3]=input_activation8
              if(input_activation8<min_mem[3]):
               min_mem[3] = input_activation8
                
              if(input_activation6>max_mem[4]):
               max_mem[4]=input_activation6
              if(input_activation6<min_mem[4]):
               min_mem[4] = input_activation6
               
              if(input_activation4>max_mem[5]):
               max_mem[5]=input_activation4
              if(input_activation4<min_mem[5]):
               min_mem[5] = input_activation4
               
               
              
              
              
              tb.add_row(["\n"+activation_layer,"\n/","\n"+str(output_bytes_convolution/1000)+"KB",str(output_bytes_convolution/1000)+"KB"]) 
              tb.add_row(["\nActivation fonction is "+activation_layer_name," "," ","This operation doesn't change the size"])
              tb.add_row([" "," "," "," "])
              tb.add_row([" "," ","size of data:"+"32bits  "+str(output_bytes_convolution*8/1000)+"Kbit","size of data:"+"32bits  "+str(output_bytes_convolution*8/1000)+"Kbit"])
              tb.add_row([" "," ","             "+"16bits  "+str(output_bytes_convolution*4/1000)+"Kbit","size of data:"+"16bits  "+str(output_bytes_convolution*4/1000)+"Kbit"])
              tb.add_row([" "," ","             "+"8bits  "+str(output_bytes_convolution*2/1000)+"Kbit","size of data:"+"8bits  "+str(output_bytes_convolution*2/1000)+"Kbit"])
              tb.add_row([" "," ","             "+"6bits  "+str(output_bytes_convolution*1.5/1000)+"Kbit","size of data:"+"6bits  "+str(output_bytes_convolution*1.5/1000)+"Kbit"])
              tb.add_row([" "," ","             "+"4bits  "+str(output_bytes_convolution/1000)+"Kbit","size of data:"+"4bits  "+str(output_bytes_convolution/1000)+"Kbit"])
              tb.add_row(["----------------------------","------------------------------------------------------------","---------------------------------------------------------","------------------------------------------"])
              
           if("pool" in tf.get_default_graph().get_operations()[a4+2].name): 
              pooling_layer_nameindex = tf.get_default_graph().get_operations()[a4+2].name.index("/")
              pooling_layer_name = tf.get_default_graph().get_operations()[a4+2].name[(pooling_layer_nameindex+1):]
              pooling_layer_nameoriginal= tf.get_default_graph().get_operations()[a4+2].name[0:pooling_layer_nameindex]
              pooling_layer = "pooling layer" #the name of pooling_layer  
              for b in memory_stats.children:              
               with open("memory_stats.txt", "w") as out:
                out.write(str(b))
               with open("memory_stats.txt", "r") as fr:
                for line1 in fr:
                 if (("name: "+"\""+pooling_layer_nameoriginal+"\"") in line1):
                  flag2=1
                  linecache.updatecache('memory_stats.txt')
                  output_bytes_pooling = int(linecache.getline('memory_stats.txt',3).split(' ')[1].strip())
                  output_size_pooling = math.sqrt(output_bytes_pooling/4/val[3])
                  output_size_pooling = int(output_size_pooling)
                  sum_memory = sum_memory+output_bytes_pooling
                  sum_memory1 = sum_memory1+output_bytes_pooling  
                  
                  list_mem0.append(output_bytes_convolution)  
                  if(output_bytes_convolution>max_mem[0]):
                    max_mem[0]=output_bytes_convolution
                  if(output_bytes_convolution<min_mem[0]):
                    min_mem[0] = output_bytes_convolution
              
                  input_pooling32=output_bytes_convolution*8/1000
                  list_mem32.append(input_pooling32)
                  input_pooling16=output_bytes_convolution*4/1000
                  list_mem16.append(input_pooling16)
                  input_pooling8=output_bytes_convolution*2/1000
                  list_mem8.append(input_pooling8)
                  input_pooling6=output_bytes_convolution*1.5/1000
                  list_mem6.append(input_pooling6)
                  input_pooling4=output_bytes_convolution*1/1000
                  list_mem4.append(input_pooling4)
                  
                  if(input_pooling32>max_mem[1]):
                    max_mem[1]=input_pooling32
                  if(input_pooling32<min_mem[1]):
                    min_mem[1] = input_pooling32
                        
                  if(input_pooling16>max_mem[2]):
                    max_mem[2]=input_pooling16
                  if(input_pooling16<min_mem[2]):
                    min_mem[2] = input_pooling16
                    
                  if(input_pooling8>max_mem[3]):
                    max_mem[3]=input_pooling8
                  if(input_pooling8<min_mem[3]):
                    min_mem[3] = input_pooling8
                    
                  if(input_pooling6>max_mem[4]):
                    max_mem[4]=input_pooling6
                  if(input_pooling6<min_mem[4]):
                    min_mem[4] = input_pooling6
                         
                  if(input_pooling4>max_mem[5]):
                    max_mem[5]=input_pooling4
                  if(input_pooling4<min_mem[5]):
                    min_mem[5] = input_pooling4  
                  
                  tb.add_row(["\n"+pooling_layer,"\n/",str(output_bytes_convolution/1000)+"KB",str(output_bytes_pooling/1000)+"KB"])
                  tb.add_row(["\nPooling fonction is "+pooling_layer_name," ","Input of pooling is the output of last layer","Pooling just changes the size of the\nmatrices, now there're "+str(val[3])+" matrices("+str(linecache.getline('memory_stats.txt',16).split(' ')[9].strip())+","+str(output_size_pooling)+","+str(output_size_pooling)+")"])
                  tb.add_row([" "," "," "," "])
                  tb.add_row([" "," ","size of data:"+"32bits  "+str(output_bytes_convolution*8/1000)+"Kbit","size of data:"+"32bits  "+str(output_bytes_pooling*8/1000)+"Kbit"])
                  tb.add_row([" "," ","             "+"16bits  "+str(output_bytes_convolution*4/1000)+"Kbit","size of data:"+"16bits  "+str(output_bytes_pooling*4/1000)+"Kbit"])
                  tb.add_row([" "," ","             "+"8bits  "+str(output_bytes_convolution*2/1000)+"Kbit","size of data:"+"8bits  "+str(output_bytes_pooling*2/1000)+"Kbit"])
                  tb.add_row([" "," ","             "+"6bits  "+str(output_bytes_convolution*1.5/1000)+"Kbit","size of data:"+"6bits  "+str(output_bytes_pooling*1.5/1000)+"Kbit"])
                  tb.add_row([" "," ","             "+"4bits  "+str(output_bytes_convolution/1000)+"Kbit","size of data:"+"4bits  "+str(output_bytes_pooling/1000)+"Kbit"])
                  tb.add_row(["----------------------------","------------------------------------------------------------","---------------------------------------------------------","------------------------------------------"])
                  break
               if(flag2==1):
                flag2=0
                break 
                
           if("pool" in tf.get_default_graph().get_operations()[a4+1].name): 
              pooling_layer_nameindex = tf.get_default_graph().get_operations()[a4+1].name.index("/")
              pooling_layer_name = tf.get_default_graph().get_operations()[a4+1].name[(pooling_layer_nameindex+1):]
              pooling_layer_nameoriginal= tf.get_default_graph().get_operations()[a4+1].name[0:pooling_layer_nameindex]
              pooling_layer = "pooling layer" #the name of pooling_layer  
              for b in memory_stats.children:              
               with open("memory_stats.txt", "w") as out:
                out.write(str(b))
               with open("memory_stats.txt", "r") as fr:
                for line1 in fr:
                 if (("name: "+"\""+pooling_layer_nameoriginal+"\"") in line1):
                  flag2=1
                  linecache.updatecache('memory_stats.txt')
                  output_bytes_pooling = int(linecache.getline('memory_stats.txt',3).split(' ')[1].strip())
                  output_size_pooling = math.sqrt(output_bytes_pooling/4/val[3])
                  output_size_pooling = int(output_size_pooling)
                  sum_memory = sum_memory+output_bytes_pooling
                  sum_memory1 = sum_memory1+output_bytes_pooling
                  
                  list_mem0.append(output_bytes_convolution)  
                  if(output_bytes_convolution>max_mem[0]):
                    max_mem[0]=output_bytes_convolution
                  if(output_bytes_convolution<min_mem[0]):
                    min_mem[0] = output_bytes_convolution
              
                  input_pooling32=output_bytes_convolution*8/1000
                  list_mem32.append(input_pooling32)
                  input_pooling16=output_bytes_convolution*4/1000
                  list_mem16.append(input_pooling16)
                  input_pooling8=output_bytes_convolution*2/1000
                  list_mem8.append(input_pooling8)
                  input_pooling6=output_bytes_convolution*1.5/1000
                  list_mem6.append(input_pooling6)
                  input_pooling4=output_bytes_convolution*1/1000
                  list_mem4.append(input_pooling4)
                  
                  if(input_pooling32>max_mem[1]):
                    max_mem[1]=input_pooling32
                  if(input_pooling32<min_mem[1]):
                    min_mem[1] = input_pooling32
                        
                  if(input_pooling16>max_mem[2]):
                    max_mem[2]=input_pooling16
                  if(input_pooling16<min_mem[2]):
                    min_mem[2] = input_pooling16
                    
                  if(input_pooling8>max_mem[3]):
                    max_mem[3]=input_pooling8
                  if(input_pooling8<min_mem[3]):
                    min_mem[3] = input_pooling8
                    
                  if(input_pooling6>max_mem[4]):
                    max_mem[4]=input_pooling6
                  if(input_pooling6<min_mem[4]):
                    min_mem[4] = input_pooling6
                         
                  if(input_pooling4>max_mem[5]):
                    max_mem[5]=input_pooling4
                  if(input_pooling4<min_mem[5]):
                    min_mem[5] = input_pooling4  
                  
                   
                  tb.add_row(["\n"+pooling_layer,"\n/",str(output_bytes_convolution/1000)+"KB",str(output_bytes_pooling/1000)+"KB"])
                  tb.add_row(["\nPooling fonction is "+pooling_layer_name," ","Input of pooling is the output of last layer","Pooling just changes the size of the\nmatrices, now there're "+str(val[3])+" matrices("+str(linecache.getline('memory_stats.txt',16).split(' ')[9].strip())+","+str(output_size_pooling)+","+str(output_size_pooling)+")"])
                  tb.add_row([" "," "," "," "])
                  tb.add_row([" "," ","size of data:"+"32bits  "+str(output_bytes_convolution*8/1000)+"Kbit","size of data:"+"32bits  "+str(output_bytes_pooling*8/1000)+"Kbit"])
                  tb.add_row([" "," ","             "+"16bits  "+str(output_bytes_convolution*4/1000)+"Kbit","size of data:"+"16bits  "+str(output_bytes_pooling*4/1000)+"Kbit"])
                  tb.add_row([" "," ","             "+"8bits  "+str(output_bytes_convolution*2/1000)+"Kbit","size of data:"+"8bits  "+str(output_bytes_pooling*2/1000)+"Kbit"])
                  tb.add_row([" "," ","             "+"6bits  "+str(output_bytes_convolution*1.5/1000)+"Kbit","size of data:"+"6bits  "+str(output_bytes_pooling*1.5/1000)+"Kbit"])
                  tb.add_row([" "," ","             "+"4bits  "+str(output_bytes_convolution/1000)+"Kbit","size of data:"+"4bits  "+str(output_bytes_pooling/1000)+"Kbit"])
                  
                  tb.add_row(["----------------------------","------------------------------------------------------------","---------------------------------------------------------","------------------------------------------"])
                  break
               if(flag2==1):
                flag2=0
                break
             
           if((v1 in tf.get_default_graph().get_operations()[a4+2].name) or ("relu" in tf.get_default_graph().get_operations()[a4+2].name) or("dropout" in tf.get_default_graph().get_operations()[a4+2].name)or("sigmoid" in tf.get_default_graph().get_operations()[a4+2].name)  or ("tanh" in tf.get_default_graph().get_operations()[a4+2].name)):
              activation_layer_nameindex = tf.get_default_graph().get_operations()[a4+2].name.index("/")
              activation_layer_name = tf.get_default_graph().get_operations()[a4+2].name[(activation_layer_nameindex+1):]
              activation_layer= "activation layer"#the name of activation_layer 
              sum_memory = sum_memory + output_bytes_pooling
              sum_memory1 = sum_memory1+output_bytes_pooling
              
              list_mem0.append(output_bytes_pooling)  
              if(output_bytes_pooling>max_mem[0]):
                max_mem[0]=output_bytes_pooling
              if(output_bytes_pooling<min_mem[0]):
                min_mem[0] = output_bytes_pooling
              
              input_pooling32=output_bytes_pooling*8/1000
              list_mem32.append(input_pooling32)
              input_pooling16=output_bytes_pooling*4/1000
              list_mem16.append(input_pooling16)
              input_pooling8=output_bytes_pooling*2/1000
              list_mem8.append(input_pooling8)
              input_pooling6=output_bytes_pooling*1.5/100
              list_mem6.append(input_pooling6)
              input_pooling4=output_bytes_pooling*1/1000
              list_mem4.append(input_pooling4)
                  
              if(input_pooling32>max_mem[1]):
                max_mem[1]=input_pooling32
              if(input_pooling32<min_mem[1]):
                min_mem[1] = input_pooling32
                    
              if(input_pooling16>max_mem[2]):
                max_mem[2]=input_pooling16
              if(input_pooling16<min_mem[2]):
                min_mem[2] = input_pooling16
                    
              if(input_pooling8>max_mem[3]):
                max_mem[3]=input_pooling8
              if(input_pooling8<min_mem[3]):
                min_mem[3] = input_pooling8
                    
              if(input_pooling6>max_mem[4]):
                max_mem[4]=input_pooling6
              if(input_pooling6<min_mem[4]):
                min_mem[4] = input_pooling6
                    
              if(input_pooling4>max_mem[5]):
                max_mem[5]=input_pooling4
              if(input_pooling4<min_mem[5]):
                min_mem[5] = input_pooling4            
              
              tb.add_row(["\n"+activation_layer,"\n/","\n"+str(output_bytes_pooling/1000)+"KB",str(output_bytes_pooling/1000)+"KB"]) 
              tb.add_row(["\nActivation fonction is "+activation_layer_name," "," ","This operation doesn't change the size"])
              tb.add_row([" "," "," "," "])
              tb.add_row([" "," ","size of data:"+"32bits  "+str(output_bytes_pooling*8/1000)+"Kbit","size of data:"+"32bits  "+str(output_bytes_pooling*8/1000)+"Kbit"])
              tb.add_row([" "," ","             "+"16bits  "+str(output_bytes_pooling*4/1000)+"Kbit","size of data:"+"16bits  "+str(output_bytes_pooling*4/1000)+"Kbit"])
              tb.add_row([" "," ","             "+"8bits  "+str(output_bytes_pooling*2/1000)+"Kbit","size of data:"+"8bits  "+str(output_bytes_pooling*2/1000)+"Kbit"])
              tb.add_row([" "," ","             "+"6bits  "+str(output_bytes_pooling*1.5/1000)+"Kbit","size of data:"+"6bits  "+str(output_bytes_pooling*1.5/1000)+"Kbit"])
              tb.add_row([" "," ","             "+"4bits  "+str(output_bytes_pooling/1000)+"Kbit","size of data:"+"4bits  "+str(output_bytes_pooling/1000)+"Kbit"])
              tb.add_row(["----------------------------","------------------------------------------------------------","---------------------------------------------------------","------------------------------------------"])
                 
           a4=0 
       if(flag1==1):
         break     
      flag1=0
    else: #dense layer (who has only 2 parts in its shape)
     count_dense = count_dense+1
     val = v.get_shape().as_list() # preserve the shape to simplify the code, it can show the shape of the layer
     for a in memory_stats.children:
       with open("memory_stats.txt", "w") as out:
        out.write(str(a))
       with open("memory_stats.txt", "r") as fr:
        for line in fr:
         if (("name: "+"\""+v1+"\"") in line):
           linecache.updatecache('memory_stats.txt')
           input_bytes_dense= (int(linecache.getline('memory_stats.txt',16).split(' ')[9].strip()))*(int(linecache.getline('memory_stats.txt',19).split(' ')[9].strip()))*4+val[0]*val[1]*4
           output_bytes_dense= int(linecache.getline('memory_stats.txt',3).split(' ')[1].strip())
           sum_memory = sum_memory + output_bytes_dense + val[0]*val[1]*4
           sum_memory1 = sum_memory1 + output_bytes_dense + val[0]*val[1]*4
    
           tb.add_row(["       ","       ",str(input_bytes_dense/1000)+"KB",str(output_bytes_dense/1000)+"KB"]) 
           if(count_conv!=(count-count_dense)):
            tb.add_row([v1,val,"Reshape matrix(2d) from last layer \n+filter of full connected layer= "+str(linecache.getline('memory_stats.txt',16).split(' ')[9].strip())+"x"+str(linecache.getline('memory_stats.txt',19).split(' ')[9].strip())+"x"+str(4)+"+"+str(val[0])+"x"+str(val[1])+"x"+str(4)+" = "+str(input_bytes_dense),"We can see there are(is) "+str(output_bytes_dense//40)+" prediction\nresult(s) in total by divide by 40"])
            
            list_mem0.append(input_bytes_dense)  
            if(input_bytes_dense>max_mem[0]):
              max_mem[0]=input_bytes_dense
            if(input_bytes_dense<min_mem[0]):
              min_mem[0] = input_bytes_dense
              
            input_dense32=input_bytes_dense*8/1000
            list_mem32.append(input_dense32)
            input_dense16=input_bytes_dense*4/1000
            list_mem16.append(input_dense16)
            input_dense8=input_bytes_dense*2/1000
            list_mem8.append(input_dense8)
            input_dense6=input_bytes_dense*1.5/100
            list_mem6.append(input_dense6)
            input_dense4=input_bytes_dense*1/1000
            list_mem4.append(input_dense4)
                  
            if(input_dense32>max_mem[1]):
              max_mem[1]=input_dense32
            if(input_dense32<min_mem[1]):
              min_mem[1] = input_dense32
                   
            if(input_dense16>max_mem[2]):
              max_mem[2]=input_dense16
            if(input_dense16<min_mem[2]):
              min_mem[2] = input_dense16
                    
            if(input_dense8>max_mem[3]):
              max_mem[3]=input_dense8
            if(input_dense8<min_mem[3]):
              min_mem[3] = input_dense8
                    
            if(input_dense6>max_mem[4]):
              max_mem[4]=input_dense6
            if(input_dense6<min_mem[4]):
              min_mem[4] = input_dense6
                    
            if(input_dense4>max_mem[5]):
              max_mem[5]=input_dense4
            if(input_dense4<min_mem[5]):
              min_mem[5] = input_dense4    
            
           else:
            tb.add_row([v1,val,"Reshape matrix(2d) from last layer \n+filter of full connected layer= "+str(linecache.getline('memory_stats.txt',16).split(' ')[9].strip())+"x"+str(linecache.getline('memory_stats.txt',19).split(' ')[9].strip())+"x"+str(4)+"+"+str(val[0])+"x"+str(val[1])+"x"+str(4)+" = "+str(input_bytes_dense),"Multiple FC layers \ncan extract the features better"])
            tb.add_row([" "," "," "," "])
           
            list_mem0.append(input_bytes_dense)  
            if(input_bytes_dense>max_mem[0]):
              max_mem[0]=input_bytes_dense
            if(input_bytes_dense<min_mem[0]):
              min_mem[0] = input_bytes_dense
              
            list_mem0.append(output_bytes_dense)  
            if(output_bytes_dense>max_mem[0]):
              max_mem[0]=output_bytes_dense
            if(output_bytes_dense<min_mem[0]):
              min_mem[0] = output_bytes_dense  
              
            input_dense32=input_bytes_dense*8/1000
            list_mem32.append(input_dense32)
            input_dense16=input_bytes_dense*4/1000
            list_mem16.append(input_dense16)
            input_dense8=input_bytes_dense*2/1000
            list_mem8.append(input_dense8)
            input_dense6=input_bytes_dense*1.5/100
            list_mem6.append(input_dense6)
            input_dense4=input_bytes_dense*1/1000
            list_mem4.append(input_dense4)
                  
            if(input_dense32>max_mem[1]):
              max_mem[1]=input_dense32
            if(input_dense32<min_mem[1]):
              min_mem[1] = input_dense32
                   
            if(input_dense16>max_mem[2]):
              max_mem[2]=input_dense16
            if(input_dense16<min_mem[2]):
              min_mem[2] = input_dense16
                    
            if(input_dense8>max_mem[3]):
              max_mem[3]=input_dense8
            if(input_dense8<min_mem[3]):
              min_mem[3] = input_dense8
                    
            if(input_dense6>max_mem[4]):
              max_mem[4]=input_dense6
            if(input_dense6<min_mem[4]):
              min_mem[4] = input_dense6
                    
            if(input_dense4>max_mem[5]):
              max_mem[5]=input_dense4
            if(input_dense4<min_mem[5]):
              min_mem[5] = input_dense4 
              
           output_dense32=output_bytes_dense*8/1000
           list_mem32.append(output_dense32)
           output_dense16=output_bytes_dense*4/1000
           list_mem16.append(output_dense16)
           output_dense8=output_bytes_dense*2/1000
           list_mem8.append(output_dense8)
           output_dense6=output_bytes_dense*1.5/1000
           list_mem6.append(output_dense6)
           output_dense4=output_bytes_dense*1/1000    
           list_mem4.append(output_dense4)
           
           if(output_dense32>max_mem[1]):
              max_mem[1]=output_dense32
           if(output_dense32<min_mem[1]):
              min_mem[1] = output_dense32
                   
           if(output_dense16>max_mem[2]):
             max_mem[2]=output_dense16
           if(output_dense16<min_mem[2]):
             min_mem[2] = output_dense16
                    
           if(output_dense8>max_mem[3]):
             max_mem[3]=output_dense8
           if(output_dense8<min_mem[3]):
             min_mem[3] = output_dense8
                    
           if(output_dense6>max_mem[4]):
             max_mem[4]=output_dense6
           if(output_dense6<min_mem[4]):
             min_mem[4] = output_dense6
                    
           if(output_dense4>max_mem[5]):
             max_mem[5]=output_dense4
           if(output_dense4<min_mem[5]):
             min_mem[5] = output_dense4 
           

           tb.add_row([" "," ","size of data:"+"32bits  "+str(input_bytes_dense*8/1000)+"Kbit","size of data:"+"32bits  "+str(output_bytes_dense*8/1000)+"Kbit"])
           tb.add_row([" "," ","             "+"16bits  "+str(input_bytes_dense*4/1000)+"Kbit","size of data:"+"16bits  "+str(output_bytes_dense*4/1000)+"Kbit"])
           tb.add_row([" "," ","             "+"8bits  "+str(input_bytes_dense*2/1000)+"Kbit","size of data:"+"8bits  "+str(output_bytes_dense*2/1000)+"Kbit"])
           tb.add_row([" "," ","             "+"6bits  "+str(input_bytes_dense*1.5/1000)+"Kbit","size of data:"+"6bits  "+str(output_bytes_pooling*1.5/1000)+"Kbit"])
           tb.add_row([" "," ","             "+"4bits  "+str(input_bytes_dense/1000)+"Kbit","size of data:"+"4bits  "+str(output_bytes_dense/1000)+"Kbit"])
           tb.add_row(["----------------------------","------------------------------------------------------------","---------------------------------------------------------","------------------------------------------"])
           
           flag3=1
           break
       if(flag3==1):
        flag3=0
        break
 x=-1
 y=0
 list_num=['size of data:32 bit(type KByte)','size of data:32 bit(type Kb)','size of data:16 bit(type Kb)','size of data:8 bit(type Kb)','size of data:6 bit(type Kb)','size of data:4 bit(type Kb)'] 
 for v in range(0,6):
  tb1=pt.PrettyTable()
  tb1.field_names = ["Range of consumption by bloc","Specific statistics"]
  tb1.align='l'
  x=x+1
  y=y+1
  range_mem = 0 
  range_mem = (max_mem[v]-min_mem[v])/10
  list_mem1=list_mem[x:y]
  list_mem2=list_mem1[0]
  list_mem3=list_mem2[:]
  b=0
  if(v==0):
   with open("memory_stats:data 1KB.txt","w") as out:
    out.write(list_num[v]+"\n")
    length = len(list_mem2)
    
    for a in range(0,10):
      tb1.add_row([str(round((min_mem[v]/1000),3))+"<consumption<="+str(round(((min_mem[v]+range_mem)/1000),3))," "])
      min_mem[v] = min_mem[v]+range_mem
      b=0
      while (b<length):
       e = list_mem2[b]
       c = list_mem2.count(e)
       if(e<=(min_mem[v]+0.005)):
        tb1.add_row([" ","---"+str(round((e/1000),3))+":"+str(c)])
        for d in range(0,length):
         if(list_mem3[d]==e):
          del list_mem2[b]
        list_mem3=list_mem2[:]
        b=-1
       else:
        c=0
       b=b+1
       length = length-c
    out.write("Stats from the obtained results\n")
    out.write(str(tb1))
    out.write("\n\n")
 
 
  if(v==1):
   with open("memory_stats:data 32bits.txt","w") as out:
    out.write(list_num[v]+"\n")
    length = len(list_mem2)
    
    for a in range(0,10):
      tb1.add_row([str(round(min_mem[v],3))+"<consumption<="+str(round((min_mem[v]+range_mem),3))," "])
      min_mem[v] = min_mem[v]+range_mem
      b=0
      while (b<length):
       e = list_mem2[b]
       c = list_mem2.count(e)
       if(e<=(min_mem[v]+0.005)):
        tb1.add_row([" ","---"+str(round((e),3))+":"+str(c)])
        for d in range(0,length):
         if(list_mem3[d]==e):
          del list_mem2[b]
        list_mem3=list_mem2[:]
        b=-1
       else:
        c=0
       b=b+1
       length = length-c
    out.write("Stats from the obtained results\n")
    out.write(str(tb1))
    out.write("\n\n")
  
  if(v==2):
   with open("memory_stats:data 16bits.txt","w") as out:
    out.write(list_num[v]+"\n")
    length = len(list_mem2)
    
    for a in range(0,10):
      tb1.add_row([str(round(min_mem[v],3))+"<consumption<="+str(round((min_mem[v]+range_mem),3))," "])
      min_mem[v] = min_mem[v]+range_mem
      b=0
      while (b<length):
       e = list_mem2[b]
       c = list_mem2.count(e)
       if(e<=(min_mem[v]+0.005)):
        tb1.add_row([" ","---"+str(round((e),3))+":"+str(c)])
        for d in range(0,length):
         if(list_mem3[d]==e):
          del list_mem2[b]
        list_mem3=list_mem2[:]
        b=-1
       else:
        c=0
       b=b+1
       length = length-c
    out.write("Stats from the obtained results\n")
    out.write(str(tb1))
    out.write("\n\n")   
    
  if(v==3):
   with open("memory_stats:data 8bits.txt","w") as out:
    out.write(list_num[v]+"\n")
    length = len(list_mem2)
    
    for a in range(0,10):
      tb1.add_row([str(round(min_mem[v],3))+"<consumption<="+str(round((min_mem[v]+range_mem),3))," "])
      min_mem[v] = min_mem[v]+range_mem
      b=0
      while (b<length):
       e = list_mem2[b]
       c = list_mem2.count(e)
       if(e<=(min_mem[v]+0.005)):
        tb1.add_row([" ","---"+str(round((e),3))+":"+str(c)])
        for d in range(0,length):
         if(list_mem3[d]==e):
          del list_mem2[b]
        list_mem3=list_mem2[:]
        b=-1
       else:
        c=0
       b=b+1
       length = length-c
    out.write("Stats from the obtained results\n")
    out.write(str(tb1))
    out.write("\n\n")  
    
  if(v==4):
   with open("memory_stats:data 6bits.txt","w") as out:
    out.write(list_num[v]+"\n")
    length = len(list_mem2)
    
    for a in range(0,10):
      tb1.add_row([str(round(min_mem[v],3))+"<consumption<="+str(round((min_mem[v]+range_mem),3))," "])
      min_mem[v] = min_mem[v]+range_mem
      b=0
      while (b<length):
       e = list_mem2[b]
       c = list_mem2.count(e)
       if(e<=(min_mem[v]+0.005)):
        tb1.add_row([" ","---"+str(round((e),3))+":"+str(c)])
        for d in range(0,length):
         if(list_mem3[d]==e):
          del list_mem2[b]
        list_mem3=list_mem2[:]
        b=-1
       else:
        c=0
       b=b+1
       length = length-c
    out.write("Stats from the obtained results\n")
    out.write(str(tb1))
    out.write("\n\n")    
    
    
  if(v==5):
   with open("memory_stats:data 4bits.txt","w") as out:
    out.write(list_num[v]+"\n")
    length = len(list_mem2)
    
    for a in range(0,10):
      tb1.add_row([str(round(min_mem[v],3))+"<consumption<="+str(round((min_mem[v]+range_mem),3))," "])
      min_mem[v] = min_mem[v]+range_mem
      b=0
      while (b<length):
       e = list_mem2[b]
       c = list_mem2.count(e)
       if(e<=(min_mem[v]+0.005)):
        tb1.add_row([" ","---"+str(round((e),3))+":"+str(c)])
        for d in range(0,length):
         if(list_mem3[d]==e):
          del list_mem2[b]
        list_mem3=list_mem2[:]
        b=-1
       else:
        c=0
       b=b+1
       length = length-c
    out.write("Stats from the obtained results\n")
    out.write(str(tb1))
    out.write("\n\n")
    
 with open("memory_stats.txt","w") as out:
  out.write("\nStage Opérationnel - YU Xuanlong\nMesurer l'occupation mémoire et structure de réseaux de neurones convolutifs​ ​avec Tensorflow\nMeasure the structure and the memory consumption of CNN with Tensorflow\n\n") # title
 with open("memory_stats.txt","a") as out: # append write
   out.write("Sructure and memory consumption of CNN\n")
   out.write(str(tb))  # print the table 
   out.write("\nTotal requested bytes is " + str(sum_memory/1000) + "KB"+"\n(If your put activation functions into your convolution layers, it's no need to use RAM to stock the outputs bytes of activation functions. In this case, we have the total requested bytes: "+str(sum_memory1/1000)+"KB \nIf two values are the same, it means the activation function part(s) is(are) separated from convolution layer(s) )\n\n")  
 sys.exit(0)
 
 
########################################################################################################__________________________________________________
																			
