import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.junit.experimental.theories.Theories;

public class kmeans {
	public static final ArrayList<String[]> Centroids2 = new ArrayList<>();
	public static final ArrayList<String[]> Centroids1 = new ArrayList<>();
	public static int check = 5;
    public static class Map extends Mapper
        <Object, Text, Text, Text> {
            private Text word = new Text();
            private Text word1 = new Text();
            public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            	
                String line = value.toString();
                StringTokenizer tokenizer = new StringTokenizer(line);
                String[] x_y_map_input = new String[2];
                while (tokenizer.hasMoreTokens()) {
                	x_y_map_input = tokenizer.nextToken().split(",");
                }
                double[] distance = new double[Centroids1.size()];
                for(int i=0;i<Centroids1.size();i++) {
                	distance[i] = computedistance(Centroids1.get(i),x_y_map_input);
                }
                
                int index = findmindistance(distance);
                word =  new Text(index+","+Centroids1.get(index)[0]+","+Centroids1.get(index)[1]);
                word1 = new Text(x_y_map_input[0]+","+x_y_map_input[1]);
                
                context.write(word,word1);
        
            }
            public int findmindistance(double[] s) {
            	int min_ind =0;
            	double min_value = s[0];
            	for(int i= 0 ;i<s.length;i++) {
            		if(s[i]<min_value) {
            			min_ind = i;
            			min_value = s[i];
            		}
            	}
            	return min_ind;
            }
            
            public double computedistance(String[] a, String[] b) {
            	return Math.sqrt(Math.pow(Double.parseDouble(a[0])-Double.parseDouble(b[0]),(double)2) + Math.pow(Double.parseDouble(a[1])-Double.parseDouble(b[1]),(double)2));
            }
        }
    
    
    public static class Reduce extends Reducer
        <Text, Text, Text, Text> {
            public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            	String temp = key.toString(); 
            	String[] rt = temp.split(",");            	
            	int sumx = 0;
            	int sumy = 0;
            	int n = 0;
            	ArrayList<String[]> store = new ArrayList<>();
            	for(Text val:values ) {
            		String temp1 = val.toString();
            		String[] temp2 = temp1.split(",");
            		store.add(temp2);
            		sumx = sumx+Integer.parseInt(temp2[0]);
            		sumy = sumy+Integer.parseInt(temp2[1]);
            		n++;
            	}
            	int centroid_x = (int)sumx/n;
            	int centroid_y = (int)sumy/n;
            	String[] ncn = new String[2];
            	ncn[0] = Integer.toString(centroid_x);
            	ncn[1] = Integer.toString(centroid_y);
            	
            	double[] distance = new double[store.size()];
                for(int i=0;i<store.size();i++) {
                	distance[i] = computedistance(ncn,store.get(i));
                }
                int index = findmindistance(distance);
            	
            	if(!checkequal(store.get(index),Integer.parseInt(rt[0]) )) {
            		Centroids1.set(Integer.parseInt(rt[0]),store.get(index));
            	}
            	Text word =  new Text(Centroids1.get(Integer.parseInt(rt[0]))[0]+","+Centroids1.get(Integer.parseInt(rt[0]))[1]);
            	context.write(key,word);
            }
            public double computedistance(String[] a, String[] b) {
            	return Math.sqrt(Math.pow(Double.parseDouble(a[0])-Double.parseDouble(b[0]),(double)2) + Math.pow(Double.parseDouble(a[1])-Double.parseDouble(b[1]),(double)2));
            }
            public int findmindistance(double[] s) {
            	int min_ind =0;
            	double min_value = s[0];
            	for(int i= 0 ;i<s.length;i++) {
            		if(s[i]<min_value) {
            			min_ind = i;
            			min_value = s[i];
            		}
            	}
            	return min_ind;
            }
        }
    	public static boolean checkequal(String[] ncn,int i) {
    		if(Centroids1.get(i)[0].equalsIgnoreCase(ncn[0]) && Centroids1.get(i)[1].equalsIgnoreCase(ncn[1])) {
    			return true;
    		}
    		check++;
    		return false;
    	}
    
    	public static boolean containsornot(int[] index,int temp) {
    		for(int i = 0 ;i<index.length;i++) {
    			if(index[i]==temp) {
    				return true;
    			}
    		}
    		return false;
    	}
        public static void main(String[] args) throws Exception {
        	
        	FileReader fileReader = new FileReader(args[3]+"/data.txt");
        	BufferedReader bufferedReader = new BufferedReader(fileReader);
        	ArrayList<String[]> data_points= new ArrayList<>(); 
        	String line;
        	while((line = bufferedReader.readLine()) != null) {
        		data_points.add(line.split(","));
        	}
        	bufferedReader.close();
        	
        	int num_of_clusters = Integer.parseInt(args[2]);
        	int[] index = new int[num_of_clusters];
        	for(int i=0;i<num_of_clusters;i++) {
        		index[i] = data_points.size();
        	}
        	for(int i=0;i<num_of_clusters;i++) {
        		Centroids1.add(data_points.get(i));
        		System.out.println(data_points.get(i)[0]+data_points.get(i)[1]);
        	}
        	
        	        	
        	Configuration conf = new Configuration();
        	FileSystem fs = FileSystem.get(conf);
        	
        	while(check!=0){
        		System.out.println(check);
        		check = 0;
        		run(conf,args,fs);
        		
        	}
            
        }
        private static void run(Configuration conf,String[] args,FileSystem fs) throws Exception{
        	Job job = Job.getInstance(conf);
            job.setJarByClass(kmeans.class);
            job.setJobName("K-Means Classifier");
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);
            job.setMapperClass(kmeans.Map.class);
            job.setCombinerClass(kmeans.Reduce.class);
            job.setReducerClass(kmeans.Reduce.class);
    		Path input = new Path(args[0]);
    		Path output = new Path(args[1]);
    		if(!fs.exists(input)) 
    		{
    			System.err.println("Input file doesn't exist");
    			return;
    		}
    		if(fs.exists(output)) 
    		{
    			fs.delete(output, true);
    			System.err.println("Output file deleted");
    		}
    		FileInputFormat.addInputPath(job, input);
            FileOutputFormat.setOutputPath(job, output);
            job.waitForCompletion(true);
        }
}
