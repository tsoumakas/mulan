package mulan.sampling;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.transformation.COCOA;
import mulan.classifier.transformation.EnsembleOfClassifierChains;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.MacroAUC;
import mulan.evaluation.measure.MacroFMeasure;
import mulan.evaluation.measure.Measure;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.supervised.instance.SpreadSubsample;

public class TestCode {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		try{
			f7();
			
			//f6();
			//f5();//SpreadSubsample 对多类进行采样:
			//(5,7,12) 1.0采样(5,5,5) 2.0采样(5,7,10) 0.5采样(5,2,2)
			
			//f4(); //Collections.shuffle
			//f3();  //ArrayList get 返回的是引用
			
			
			String s="emotions";
			String filePath="F://刘彬学校电脑资料//希腊//数据//MutliLabel Datasets//";
			String fileArff=filePath+s+"//"+s+".arff";
			String fileXml=filePath+s+"//"+s+".xml";
			MultiLabelInstances mldata=new MultiLabelInstances(fileArff, fileXml);
			
			Instances ins=mldata.getDataSet();
			//ins.clear();  			
			//System.out.println(mldata.getDataSet().numInstances()+"\t"+ins.numInstances()+"\t"+mldata.getDataSet().hashCode()+"\t"+ins.hashCode());  
			//0	0  	1	1 返回的是引用,ins指向的是mldata中的Instances
			
			//f2(ins);
			//System.out.println(mldata.getDataSet().numInstances()+"\t"+mldata.getDataSet().hashCode());
			//0	1  	
			//0	1 函数传递参数时也是传递的引用。。都指向同一个Instances对象

		}
		catch (Exception e){
			e.printStackTrace();
		}
		
		
		/*
		A aa[]={new A(),new B()};
		for(A a:aa){
			a.f();
		}
		*/
		
		/*
		Random r1 = new Random(1);
		Random r2 = new Random(1);
		
		for(int i=0;i<5;i++){
			System.out.println(r1.nextInt());
			System.out.println(r2.nextInt());
		}
		
		int a[]=f1();
		int b[]=a;
		int c[]=f1();
		int d[]= new int [a.length];
		System.arraycopy(a, 0, d, 0, a.length);
		
		System.out.println(a.hashCode());
		System.out.println(b.hashCode());
		System.out.println(c.hashCode());
		System.out.println(d.hashCode());
		*/
	}
	
	
	public static void f2(Instances ins){
		ins.clear();
		System.out.println(ins.numInstances()+"\t"+ins.hashCode());  
	}
	
	public static int[] f1(){
		int a[]= {1,2,3};
		return a;
	}
	
	//ArrayList get 返回的是引用
	public static void f3(){
		ArrayList<Double> list1 = new ArrayList<Double>(2);  
        ArrayList<Double> list2 = new ArrayList<Double>(4);  
        ArrayList<Double> list3 = new ArrayList<Double>(3);  
        ArrayList<ArrayList<Double>> twoDimList = new ArrayList<ArrayList<Double>>(3);  
          
        list1.add(3.5);  
        list1.add(6.9);  
  
        list2.add(1.5);  
        list2.add(7.9);  
        list2.add(3.3);  
        list2.add(8.9);  
  
        list3.add(1.9);  
        list3.add(8.2);  
        list3.add(5.8);  
          
        twoDimList.add(list1);  
        twoDimList.add(list2);  
        twoDimList.add(list3);  
          
        int count = 0;  
        for(List<Double> tmp : twoDimList){  
            System.out.print("twoDimList index: ");  
            System.out.println(count++);  
            for(Double temp : tmp){  
                System.out.println(temp);  
            }  
        }  
          
        ArrayList<Double> tmpList = twoDimList.get(1);  
          
        System.out.println("twoDimList 2:");  
        for(Double temp : tmpList){  
            System.out.println(temp);  
        }  
          
        System.out.println("。。。Test ArrayList reference。。。！");  
          
        tmpList.remove(0);  
        tmpList.add(15.7);  
          
        count = 0;  
        for(List<Double> tmp : twoDimList){  
            System.out.print("twoDimList index: ");  
            System.out.println(count++);  
            for(Double temp : tmp){  
                System.out.println(temp);  
            }  
        }  
        System.out.println("Test end！"); 
	}
	
	//Collections.shuffle  不更改Random，每次程序运行的结果都不一样;更改了Random则每次程序运行结果都一样
	public static void f4(){
		ArrayList<Integer> a=new ArrayList<>();
		ArrayList<Integer> b=new ArrayList<>();
		ArrayList<Integer> c=new ArrayList<>();
		for(int i=0;i<10;i++){
			a.add(i);
			c.add(i);
		}
		
		Random rnd1=new Random(1);
		Random rnd2=new Random(1);
		
		Collections.shuffle(a, rnd1);		
		System.out.println(a.toString());
		//Collections.shuffle(b);		
		//System.out.println(b.toString());
		Collections.shuffle(c, rnd1);	
		System.out.println(c.toString());
		
		Collections.shuffle(a, rnd2);	
		System.out.println(a.toString());
		//Collections.shuffle(b);		
		//System.out.println(b.toString());
		Collections.shuffle(c, rnd2);		
		System.out.println(c.toString());
	}

	//SpreadSubsample 对多类进行采样:(5,7,12) 1.0采样(5,5,5) 2.0采样(5,7,10) 0.5采样(5,2,2)
	public static void f5() throws Exception{
		
		Instances ins=new Instances(new BufferedReader(new FileReader("C:\\Users\\lb\\Desktop\\glass.arff")));
		ins.setClassIndex(ins.numAttributes()-1);
		
		int classIndex=ins.classIndex();
		int numClasses=ins.numClasses();
		String classNames[]=new String[numClasses];
		Enumeration<Object> en=ins.attribute(classIndex).enumerateValues();
		int i=0;
		while (en.hasMoreElements()) {
		    classNames[i++]=en.nextElement().toString();
		}
		
		int c[]=new int[numClasses];
		Arrays.fill(c, 0);
		for(Instance data:ins){
			String classValue=data.stringValue(classIndex);
			for(i=0;i<numClasses;i++)
			{
				if(classNames[i].equals(classValue)){
					c[i]++;
				}
			}
		}
		for(i=0;i<numClasses;i++)
		System.out.println(classNames[i]+"\t"+c[i]);
		
		SpreadSubsample ss=new SpreadSubsample();
		ss.setInputFormat(ins);
		ss.setDistributionSpread(0.5);
		ins=ss.useFilter(ins, ss);
		
		
		Arrays.fill(c, 0);
		for(Instance data:ins){
			String classValue=data.stringValue(classIndex);
			for(i=0;i<numClasses;i++)
			{
				if(classNames[i].equals(classValue)){
					c[i]++;
				}
			}
		}
		for(i=0;i<numClasses;i++)
		System.out.println(classNames[i]+"\t"+c[i]);
		
	}
	
	//test COCOA
	public static void f6(){
	    ArrayList<String> listArff = new ArrayList();
	    
	    //listArff.add("emotions");
	    listArff.add("cal500");
	    int numFolds = 5;
	    
	    String path="F:\\刘彬学校电脑资料\\希腊\\数据\\MutliLabel Datasets\\RemoveHighImLabel\\";
	    COCOA cocoa = new COCOA();
	    
	    EnsembleOfClassifierChains ecc=new EnsembleOfClassifierChains();
	    ecc.setUseClassiferChainUnderSampling(true);
	    ecc.setUseFmeasureOptimizationThreshold(true);
	    ecc.setSamplingPercentage(0.5);
	    
	    MultiLabelLearner ml=ecc;
	    
	    for (int i = 0; i < listArff.size(); i++) {
	      try
	      {
	        String arffFilename = path+(String)listArff.get(i)+"\\"+(String)listArff.get(i) + ".arff";
	        String xmlFilename = path+(String)listArff.get(i)+"\\"+(String)listArff.get(i) + ".xml";
	        System.out.println("Loading the dataset...");
	        MultiLabelInstances dataset = new MultiLabelInstances(arffFilename, xmlFilename);
	        
	        dataset.getDataSet().randomize(new Random(1));
	        Instances train = dataset.getDataSet().trainCV(numFolds, 1);
            MultiLabelInstances multiTrain = new MultiLabelInstances(train, dataset.getLabelsMetaData());
            Instances test = dataset.getDataSet().testCV(numFolds, 1);
            MultiLabelInstances multiTest = new MultiLabelInstances(test, dataset.getLabelsMetaData());
        	
	        
	        Evaluator eval = new Evaluator();
	        

	        ArrayList<Measure> allMeasures = new ArrayList();
	        int numLabels = dataset.getNumLabels();
	        allMeasures.add(new MacroFMeasure(numLabels));
	        allMeasures.add(new MacroAUC(numLabels));
	        
	        eval.setSeed(2);
	        MultipleEvaluation results = eval.crossValidate(ml, dataset, allMeasures, numFolds);
	        //PrintWriter resultPrintWriter = new PrintWriter("COCOA_" + (String)listArff.get(i) + "_results.txt");
	        //resultPrintWriter.print(results);
	        //resultPrintWriter.close();
	        
	        //ml.build(multiTrain);
	        //Evaluation results = eval.evaluate(ml, multiTest,allMeasures);
	        
	        System.out.println(results.toString());
	      }
	    
	      catch (Exception ex)
	      {
	        ex.printStackTrace();
	      }
	    }
	}
	
	//test String 复制 (=也是复制其内容，并不是指向同一个对象)
	public static void f7(){
		String s2;
		String s1="abc";
		s2=s1;
		s1=s1.toUpperCase();
		System.out.println(s1+"\t"+s2);
	}
	
	
	public static void fPrintInstacnesInfo(Instances ins){
		int classIndex=ins.classIndex();
		int numClasses=ins.numClasses();
		System.out.println("#Ins\t"+ins.numInstances());
		String classNames[]=new String[numClasses];
		Enumeration<Object> en=ins.attribute(classIndex).enumerateValues();
		int i=0;
		while (en.hasMoreElements()) {
		    classNames[i++]=en.nextElement().toString();
		}
		
		int c[]=new int[numClasses];
		Arrays.fill(c, 0);
		for(Instance data:ins){
			String classValue=data.stringValue(classIndex);
			for(i=0;i<numClasses;i++)
			{
				if(classNames[i].equals(classValue)){
					c[i]++;
				}
			}
		}
		for(i=0;i<numClasses;i++)
			System.out.println(classNames[i]+"\t"+c[i]);
	}
	
}

class A {
	void f(){
		System.out.println("A");
	}
}

class B extends A{
	void f(){
		System.out.println("B");
	}
}
