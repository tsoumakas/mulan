package mulan.classifier.MultiLabelHyperNetWork;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import mulan.classifier.MultiLabelLearner;
import mulan.data.MultiLabelInstances;
import weka.classifiers.Classifier;
import weka.clusterers.RandomizableClusterer;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

/**
<!-- globalinfo-start -->
* The Basic Function that used in the MLHN-C model
* <br>
<!-- globalinfo-end -->
*
* @author LB
* @version 2017.01.10
*/

public class BaseFunction {
	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		System.out.println("BaseFunction Main");
	}

	/**
	 * Returns the name of a classifier
	 * 
	 * @param classifier a classifier
	 * @return the name of the classifier  
	 **/
	public static String Get_Classifier_name(Classifier classifier) {
		String s = classifier.getClass().toString(); // s="class mulan.classifier.lazy.MLkNN"
		int Si = s.lastIndexOf(".") + 1;
		int Ei = s.length();
		return s.substring(Si, Ei); // classifierName="MLkNN" 分类器名称
	}

	/**
	 * Returns the name of a multiLabel learner
	 * 
	 * @param  classifier a multiLabel learner
	 * @return the name of the multiLabel learner
	 **/
	public static String Get_Classifier_name(MultiLabelLearner classifier) {
		String s = classifier.getClass().toString(); // s="class mulan.classifier.lazy.MLkNN"
		int Si = s.lastIndexOf(".") + 1;
		int Ei = s.length();
		return s.substring(Si, Ei); // classifierName="MLkNN" 
	}

	/**
	 * Return a rounded double value with certain number of decimals
	 * 
	 * @param x double number be processed
	 * @param n the number of decimals be saved
	 * @return the rounded value of x with n decimals
	 **/
	public static double Round(double x, int n) {
		BigDecimal b = new BigDecimal(x);
		return b.setScale(n, BigDecimal.ROUND_HALF_UP).doubleValue();
	}

	/**
	 * Returns a list of String[], each String[] contain a line in CSV file 
	 * 
	 * @param CSVFileName the name of a csv file
	 * @param split the sperator of each element in a line 
	 * @return a list of String[] contained the content of the file
	 */
	public static List<String[]> readCSV(String CSVFileName,String split){		
		List<String[]> list=null;
		
		try 
		{
			
			FileInputStream fis = new FileInputStream(CSVFileName);  
			InputStreamReader  isw = new InputStreamReader(fis, "UTF-8"); 
			BufferedReader reader = new BufferedReader(isw);
			
			list=new ArrayList<String[]>();
			String str;
	        while((str = reader.readLine())!= null)
	        {
	        	String sg[]=str.split(split);
	        	list.add(sg);
	        }
	        reader.close();
	    } 
		catch (Exception e) {
	        e.printStackTrace();
	    }
		finally{
			return list;	
		}	
	}

	/**
	 * Writes the content of List<String[]> to a CSV file
	 * 
	 * @param CSVFileName the name of a csv file
	 * @param content a list of String[] whoes content is used to write
	 * @param Is_add ture if the content is add to the end of the file, 
	 * 				 otherwise the original content of the file would be covered 
	 * @param codeType the name of a supported charset
	 */
	public static void writeCSV(String CSVFileName,List<String[]> content,boolean Is_add,String codeType){		
		try 
		{
			if(content==null||content.size()<=0){
				return;
			}
			
			FileOutputStream fos = new FileOutputStream(CSVFileName,Is_add); 
	        OutputStreamWriter osw = new OutputStreamWriter(fos, codeType); 
	        int i=0;
	        String out="";
	        for(String sg[]:content)
	        {
	        	for(String str:sg)
	        	{
	        		out+=str+",";
	        	}
	        	out=out.substring(0,out.length()-1);
	        	out+="\n";
	        	i++;
	        	if(i%100==0){
	        		 osw.write(out); 
	        		 out="";
	        	}
	        }
	        
	        osw.write(out); 
	        osw.flush();
	        
			osw.close();
			fos.close();
			
	    }
		catch (Exception e) {
	        	e.printStackTrace();
	    }
		
		
	}

	/**
	 * Writes a string to a CSV file
	 * 
	 * @param fileallname the name of a csv file
	 * @param out a string is used to write
	 * @param Is_add ture if the content is add to the end of the file, 
	 * 				 otherwise the original content of the file would be covered 
	 * @param codeType the name of a supported charset
	 */
	public static void Out_file(String fileallname,String out,boolean Is_add,String codeType){
		try{
			
			FileOutputStream fos = new FileOutputStream(fileallname,Is_add); 
	        OutputStreamWriter osw = new OutputStreamWriter(fos, codeType); 
	        osw.write(out); 
	        osw.flush();
	        
			osw.close();
			fos.close();
		}
		catch (Exception e){
			e.printStackTrace();
		}
	}
	
	public static void Out_file(String fileallname,String out,boolean Is_add){
		try{
			FileWriter fout=new FileWriter(fileallname,Is_add);
			fout.write(out);
			fout.close();
		}
		catch (Exception e){
			e.printStackTrace();
		}
	}
	
	/**
	 * Writes an instances to a file
	 * 
	 * @param instances an instances to be saved
	 * @param filename the name of file
	 * @param filepath the path of file
	 */
	public static void Outfile_instances_arff(Instances instances,String filename, String filepath){	
		//判断目录，若不存在则新建
        File file=new File(filepath);
        if  (!file .exists()  && !file .isDirectory())      
        {       
            System.out.println("//不存在"+filepath+"需创建");  
            file .mkdir();    
        }
        
		ArffSaver saver = new ArffSaver();
     	saver.setInstances(instances);
		String file_entire_name=filepath+filename;
		try{
			saver.setFile(new File(file_entire_name));
			saver.writeBatch();
		}
		catch (Exception e){
			e.printStackTrace();
		}
			 
		System.out.println(file+"写入文件");	
	}	
	
	/**
	 * Writes an instances to a file with suffix of "arff"
	 * 
	 * @param instances an instances to be saved
	 * @param filename the whole name of file(including path)
	 */
	public static void Outfile_instances_arff(Instances instances,String filename){	
        
		ArffSaver saver = new ArffSaver();
     	saver.setInstances(instances);
		String file_entire_name=filename;
		try{
			saver.setFile(new File(file_entire_name));
			saver.writeBatch();
		}
		catch (Exception e){
			e.printStackTrace();
		}
			 
		System.out.println(file_entire_name+"写入文件");	
	}	
	
	
	/**
	 * Transforms an array to an ArrayList
	 * @param obj an array  
	 * @return an ArrayList transformed by obj
	 * @throws Exception if transformation fails
	 */
    public static ArrayList getArrayList(Object[] obj) throws Exception{
    	ArrayList list = new ArrayList();
        for(int i=0;i<obj.length;i++)
            list.add(obj[i]);
        return list;
    }
    
    
    /**
     * Returns a random number in a certain range
     * 
     * @param min the minimum
     * @param max the maximum
     * @return a random number in the range of [min,max]
     */
	public static int randomInt(int min,int max){
		Random r = new Random();
		int i=Math.abs(r.nextInt())%(max-min+1)+min;
		return i;
	}
    
    /**
     * Returns an array of random number without replicated in a certain range
     * 
     * @param min the minimum
     * @param max the maximum
     * @param num the count of numbers that returns
     * @return an array of random number in the range of [min,max]
     */
	public static int[] randomIntArray(int max,int min,int num){
		int n[]=new int[num];
		int lenght=max-min+1;
		int tag[]=new int[lenght];
		
		for(int i=0;i<lenght;i++){
			tag[i]=0; //0为未被选中
		}
		int r;
		for(int i=0;i<num;i++){
			do{
				r=randomInt(min, max);
			}while(tag[r-min]==1);
			n[i]=r;
			tag[r-min]=1;
		}
		
		return n;
	}
	
	//返回一个在min和max之间的随机数组（包括min和max，不重复）
	
	public static Integer[] randomIntegerArray(int max,int min,int num){
		Integer n[]=new Integer[num];
		int lenght=max-min+1;
		int tag[]=new int[lenght];
		
		for(int i=0;i<lenght;i++){
			tag[i]=0; //0为未被选中
		}
		int r;
		for(int i=0;i<num;i++){
			do{
				r=randomInt(min, max);
			}while(tag[r-min]==1);
			n[i]=r;
			tag[r-min]=1;
		}
		
		return n;
	}
	
	//返回数组的平均值
	
	/**
	 * Returns the average of decimals in an array
	 * 
	 * @param A double array 
	 * @return the average of decimals in A
	 */
	public static double Get_Average(double A[]){
		if(A==null||A.length==0){
			return 0;
		}
		
		double d=0.0;
		for(int i=0;i<A.length;i++){
			d+=A[i];
		}
		return d/A.length;
	}
		
	//返回数组的标准差
	
	/**
	 * Returns the standard deviation of decimals in an array
	 * 
	 * @param A double array 
	 * @return the standard deviation of decimals in A
	 */
	public static double Get_Std(double A[]){
		if(A==null||A.length==0){
			return 0;
		}
		
		double ave=0.0;
		for(int i=0;i<A.length;i++){
			ave+=A[i];
		}
		ave/=A.length;
		double result=0.0;
		
		for(double a:A){
			result+=(a-ave)*(a-ave);
		}
		result/=A.length;
		
		return Math.sqrt(result);
	}

	//返回数组的平均值
	
	public static double Get_Average(List<Double> A){
		if(A==null||A.size()==0){
			return 0;
		}
		
		double d=0.0;
		for(int i=0;i<A.size();i++){
			d+=A.get(i);
		}
		return d/A.size();
	}
		
	//返回数组的标准差
	
	public static double Get_Std(List<Double> A){
		if(A==null||A.size()==0){
			return 0;
		}
		
		double ave=Get_Average(A);
		double result=0.0;
		
		for(double a:A){
			result+=(a-ave)*(a-ave);
		}
		result/=A.size();
		
		return Math.sqrt(result);
	}


}
