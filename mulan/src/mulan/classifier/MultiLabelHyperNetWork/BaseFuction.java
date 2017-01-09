package mulan.classifier.MultiLabelHyperNetWork;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import mulan.classifier.MultiLabelLearner;
import mulan.data.MultiLabelInstances;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class BaseFuction {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
	}

	// 得到分类器名称
	public static String Get_Classifier_name(Classifier classifier) {
		String s = classifier.getClass().toString(); // s="class mulan.classifier.lazy.MLkNN"
		int Si = s.lastIndexOf(".") + 1;
		int Ei = s.length();
		return s.substring(Si, Ei); // classifierName="MLkNN" 分类器名称
	}

	// 得到分类器名称
	public static String Get_Classifier_name(MultiLabelLearner classifier) {
		String s = classifier.getClass().toString(); // s="class mulan.classifier.lazy.MLkNN"
		int Si = s.lastIndexOf(".") + 1;
		int Ei = s.length();
		return s.substring(Si, Ei); // classifierName="MLkNN" 分类器名称
	}

	// 四舍五入保留n位小数
	public static double siSheWuRu(double x, int n) {
		BigDecimal b = new BigDecimal(x);
		return b.setScale(n, BigDecimal.ROUND_HALF_UP).doubleValue();
	}

	
	//读取CSV文件，每行记录在一个String[]中，返回List<String[]>
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

	//将content中的内容写入CSV文件，
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

	
	
	//文件完全名称fileallname，out输出内容,Is_add为true时在文件末尾添加内容，为false时覆盖原先内容
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
	
	//文件完全名称fileallname，out输出内容,codeType 为编码模式
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
	
	
	
	//样本集写入文件(arff文件)
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
	
	
	//数组转ArrayList
    public static ArrayList getArrayList(Object[] obj)throws Exception{
    	ArrayList list = new ArrayList();
        for(int i=0;i<obj.length;i++)
            list.add(obj[i]);
        return list;
    }
    
    
	//返回一个在min和max之间的随机数（包括min和max）
	public static int randomInt(int min,int max){
		Random r = new Random();
		int i=Math.abs(r.nextInt())%(max-min+1)+min;
		return i;
	}
    
	//返回一个在min和max之间的随机数组（包括min和max，不重复）
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
