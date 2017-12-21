import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.hypernet.BaseFunction;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.EnsembleOfClassifierChains;
import mulan.classifier.transformation.MultiLabelStacking;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.MacroAUC;
import mulan.evaluation.measure.MacroAUCPR;
import mulan.evaluation.measure.MacroFMeasure;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.MicroAUC;
import mulan.evaluation.measure.MicroFMeasure;
import mulan.evaluation.measure.RankingLoss;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LibLINEAR;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;

public class VSDataTest {

public static String filePath="F:\\刘彬学校电脑资料\\希腊\\数据\\VS Dataset\\feturalization of compound & target\\Dataset5\\MultiLabel Dataset\\";
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
		boolean isHyperion=true;
		if(isHyperion)
			filePath="//data//mlkd//BinLiu//Dataset5//MultiLabel Dataset//";
		
		//NaiveBayes nb=new NaiveBayes();
		J48 j48=new J48();
		RandomForest rf=new RandomForest();
		LibLINEAR svm =new LibLINEAR();
		Classifier cla=j48;
		//LibSVM svm=new LibSVM();  //Use libsvm in Weka    web.facebook.com/groups/AccommodationInThessalonikiByESN/
		
		BinaryRelevance br=new BinaryRelevance(cla);
		EnsembleOfClassifierChains ecc=new EnsembleOfClassifierChains(cla, 10, true, true);
		MultiLabelStacking sta=new MultiLabelStacking();
		
		String dataNames[]={"IC50GPCR","BEIGPCR,IC50","KiGPCR","BEIGPCR,Ki","IC50PK","BEIPK,IC50"};  //
		MultiLabelLearner mls[]={br,ecc};//,sta,ecc
		String outFileName=filePath+"Result"+BaseFunction.Get_Classifier_name(cla)+".txt";
		
		
		int flod=5;
		long beginTime,endTime;
		double trainingTime[]=new double[flod];
		double testTime[]=new double[flod];
		
		StringBuffer sb=new StringBuffer();
		BaseFunction.Out_file(outFileName, "", false);
		try{
			for(String dataName:dataNames){
			    MultiLabelInstances mlData=new MultiLabelInstances(filePath+"MultiLabelDataSet "+dataName+" training data_filterNullRedundant_filterLabels.arff"
						,filePath+"MultiLabelDataSet "+dataName+" training data_filterNullRedundant_filterLabels.xml");
			    mlData.getDataSet().randomize(new Random(1));
			    
			    sb.append(dataName+"\n");
			    sb.append("#instance: "+mlData.getNumInstances()+"\t#label: "+mlData.getNumLabels()+"\tCard: "+mlData.getCardinality()+"\n");
			    
			    BaseFunction.Out_file(outFileName, sb.toString(), true);
		        System.out.print(sb.toString());
			    sb.delete(0, sb.length());
			    
		    	Evaluator eval = new Evaluator();
	            MultipleEvaluation multiEval = new MultipleEvaluation(mlData);
	            int numLabels = mlData.getNumLabels();
	            Measure evaluationMeasures[]={new HammingLoss(),new MacroFMeasure(numLabels),new MacroAUC(numLabels),new MacroAUCPR(numLabels)
	            		,new MicroFMeasure(numLabels),new MicroAUC(numLabels)};
	            //
	            
	            for(MultiLabelLearner ml:mls){
	            	//sb.append(BaseFunction.Get_Classifier_name(ml)).append("\n");
	            	System.out.println(BaseFunction.Get_Classifier_name(ml));
	            	for(int i=0;i<flod;i++){
	            		MultiLabelInstances multiTrain=new MultiLabelInstances(mlData.getDataSet().trainCV(flod, i),mlData.getLabelsMetaData());
	            		MultiLabelInstances multiTest=new MultiLabelInstances(mlData.getDataSet().testCV(flod, i),mlData.getLabelsMetaData());
	            		/*
	            		BaseFunction.Out_file(filePath+"MultiLabelDataSet "+dataName+" training data_filterNullRedundant_filterLabels_TrainCV"+i+".arff"
	            				,multiTrain.getDataSet().toString() , false);
	            		BaseFunction.Out_file(filePath+"MultiLabelDataSet "+dataName+" training data_filterNullRedundant_filterLabels_TestCV"+i+".arff"
	            				,multiTest.getDataSet().toString() , false);
	            		*/
	            		
	            		/*
	            		sb.append("CV"+i+"\t").append("Train #instance: "+multiTrain.getNumInstances()+"\t#label: "+multiTrain.getNumLabels()+"\tCard: "+multiTrain.getCardinality()+"\t")
	            		.append("Test #instance: "+multiTest.getNumInstances()+"\t#label: "+multiTest.getNumLabels()+"\tCard: "+multiTest.getCardinality()+"\n");
	            		*/
    	                System.out.print("Begin:"+df.format(System.currentTimeMillis()).toString()+"\t");
    	            	beginTime=System.currentTimeMillis();
    	                ml.build(multiTrain);
    	                endTime=System.currentTimeMillis();
    	                trainingTime[i]=(endTime-beginTime)/1000.0;
    	                
    	                
    	                Evaluator evaluator = new Evaluator();
                       
                        beginTime=System.currentTimeMillis();
                        Evaluation e = evaluator.evaluate(ml, multiTest, Arrays.asList(evaluationMeasures));
                        endTime=System.currentTimeMillis();
    	                testTime[i]=(endTime-beginTime)/1000.0;		    	                
    	                System.out.print("End:"+df.format(System.currentTimeMillis()).toString()+"\n");
    	                
                        multiEval.addEvaluation(e);
                        
			    	}
			        
	            	multiEval.calculateStatistics();
	            	
	            	sb.append(BaseFunction.Get_Classifier_name(ml)).append("\n");
	            	for(Measure m:evaluationMeasures){
	            		String measureName=m.getName();		            		
	            		sb.append(measureName+"\t"+multiEval.getMean(measureName)+"\t"+multiEval.getStd(measureName)+"\n");
	            	}
	            	sb.append("Training Time\t").append(BaseFunction.Get_Average(trainingTime)+"\t").append(BaseFunction.Get_Std(trainingTime)+"\n");
	            	sb.append("Test Time\t").append(BaseFunction.Get_Average(testTime)+"\t").append(BaseFunction.Get_Std(testTime)+"\n");
	            	sb.append("\n");
	            	System.out.println(sb);
	            	BaseFunction.Out_file(outFileName, sb.toString(), true);
	            	sb.delete(0, sb.length());	 
			        
			    }
			}
			

		}
		catch(Exception e){
			e.printStackTrace();
		}
	}
	
	
}
