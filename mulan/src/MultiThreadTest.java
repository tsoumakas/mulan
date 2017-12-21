import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.hypernet.BaseFunction;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.BinaryRelevanceUnderSampling;
import mulan.classifier.transformation.COCOA;
import mulan.classifier.transformation.EnsembleBinaryRelevanceUnderSampling;
import mulan.classifier.transformation.EnsembleOfClassifierChains;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.MacroAUC;
import mulan.evaluation.measure.MacroAUCPR;
import mulan.evaluation.measure.MacroFMeasure;
import mulan.evaluation.measure.Measure;
import mulan.sampling.ImUtil;
import mulan.sampling.MultiLabelSampling;
import mulan.sampling.MutilLabelRandomUnderSampling;
import mulan.sampling.NoProcess;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;

public class MultiThreadTest {
	public static SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
	
	public static void main(String[] args) {
		//fTestECCUSVariousModelNumber();
		testECCUS();
	}
	
	public static void testECCUS(){
		double Ps[]={0.1};
		boolean isRemoveHI=false;
		boolean isHyperion=false;
		MultiLabelSampling samplings[]={new NoProcess(),new NoProcess(),new NoProcess(),new NoProcess(),
				new MutilLabelRandomUnderSampling(),new MutilLabelRandomUnderSampling(),new NoProcess(),new NoProcess()}; //
		//ImbalancedStatistics is=new ImbalancedStatistics();
		String dataNames[]={"emotions"};//"rcv1subset1","rcv1subset2","bibtex","bookmarks","delicious","eurlex-sm","medical","enron","tmc2007-500",Corel5k,"yeast","flags","yahoo-Arts1","yahoo-Business1"
		//"enron","emotions","cal500","enron","yeast","medical","scene","flags","birds","genbase","mediamill"
		String  dataName0_1[]={"bibtex","bookmarks","delicious","enron","eurlex-sm","medical"};
		String  dataName0_01[]={"rcv1subset1","rcv1subset2","yahoo-Arts1","yahoo-Business1"};
		HashSet<String> dataName0_1Set=new HashSet<>(); dataName0_1Set.addAll(Arrays.asList(dataName0_1));
		HashSet<String> dataName0_01Set=new HashSet<>(); dataName0_01Set.addAll(Arrays.asList(dataName0_01));
				
		
		Classifier cla=new J48();// LibLINEAR()
		
		BinaryRelevance br1=new BinaryRelevance(cla);
		EnsembleOfClassifierChains ecc1=new EnsembleOfClassifierChains(cla, 10, true, true);
		BinaryRelevanceUnderSampling brus=new BinaryRelevanceUnderSampling(cla, 1.0);
		
		EnsembleBinaryRelevanceUnderSampling ebrus=new EnsembleBinaryRelevanceUnderSampling(cla);
		ebrus.setNumOfModels(10);
		ebrus.setUnderSamplingPercent(1.0);
		ebrus.setUseConfidences(true);
		ebrus.setUseFmeasureOptimizationThreshold(true);
		
		EnsembleOfClassifierChains eccus=new EnsembleOfClassifierChains(cla, 10, true, true);
		eccus.setUseClassiferChainUnderSampling(true);
		eccus.setUseFmeasureOptimizationThreshold(true);
		eccus.setUnderSamplingPercent(1.0);  //1.0: the equal size of the majority and minority instances after under sampling 

		COCOA cocoa=new COCOA(cla,10);
		cocoa.setUnderSamplingPercent(1.0);
				

		MultiLabelLearner mls[]={br1,ecc1,brus,ebrus,br1,ecc1,cocoa,eccus}; //
		
        int numRepetitions=2, numFlods=5;   
		long seeds[]=new long[numRepetitions];
		for(int i=0;i<numRepetitions;i++){
			seeds[i]=i+1L;
		}
		
        double trainingTime[]=new double[numRepetitions*numFlods];
        double testTime[]=new double[numRepetitions*numFlods];
        long beginTime,endTime;
        
        
        try{	
			for(String dataName:dataNames){
				
				System.out.println(dataName);
				
				String filePath=isHyperion?"//data//mlkd//BinLiu//MultiLabelDataSet//"+dataName+"//" : "F://刘彬学校电脑资料//希腊//数据//MutliLabel Datasets//"+(isRemoveHI?"RemoveHighImLabel":"")+"//"+dataName+"//";//
				String outFilePath=isHyperion?"//data//mlkd//BinLiu//MultiLabelDataSet//RESULT//4//" : "F://刘彬学校电脑资料//希腊//experiment result//ecc"+(isRemoveHI?"_RemoveHighImLabel":"")+"5"+"//";//  //_RemoveHighImLabel
				String arrfFile=dataName;
				String xmlFile=dataName+".xml";
				if(dataName0_1Set.contains(dataName)){
					arrfFile+="_0.1";
				}
				if(dataName0_01Set.contains(dataName)){
					arrfFile+="_0.01";
				}

				
				System.out.println("Loading the data set");
				MultiLabelInstances []mldatas=new MultiLabelInstances[numRepetitions];
				mldatas[0]=new MultiLabelInstances(filePath+arrfFile+".arff",filePath+xmlFile);
				for(int repetition=1;repetition<numRepetitions;repetition++){
					mldatas[repetition]=mldatas[0].clone();
					mldatas[repetition].getDataSet().randomize(new Random(seeds[repetition]));
				}
				MultiLabelInstances multiTrain=null;
				MultiLabelInstances multiTest=null;
				
				
	            Evaluator evaluator;
	            Measure[] evaluationMeasures = {new MacroFMeasure(mldatas[0].getNumLabels()),new MacroAUC(mldatas[0].getNumLabels()),new MacroAUCPR(mldatas[0].getNumLabels())};
	            	/*
	            	 * ,new MicroFMeasure(mldatas[0].getNumLabels()),new MicroAUC(mldatas[0].getNumLabels()),
	            	 * {new MacroFMeasure(mldata.getNumLabels()),new MacroPrecision(mldata.getNumLabels()),new MacroRecall(mldata.getNumLabels()),
	            								new MacroAUC(mldata.getNumLabels()),new MicroFMeasure(mldata.getNumLabels()),new MicroPrecision(mldata.getNumLabels()),
	            								new MicroRecall(mldata.getNumLabels()),new MicroAUC(mldata.getNumLabels())};
				*/
				String outFileResult=outFilePath+dataName+"Result.txt";
				//String outFileImSta=outFilePath+dataName;  //the imbalanced Statistics of original and processed dataset
				StringBuffer sb1=new StringBuffer();
				//StringBuffer sb2=new StringBuffer();
				BaseFunction.Out_file(outFileResult, "", false);

	            for(int i=0;i<samplings.length;i++){
	            	MultiLabelSampling sp=samplings[i];
	            	double dd[]={0};
	            	
	            	if(!sp.getClass().equals(NoProcess.class))
	            		dd=Ps;
	            	
	            	for(double P:dd){	
	            		if(sp.getClass().equals(NoProcess.class)){
		            		
	            			String s = "";
	            			if(mls[i].getClass().equals(EnsembleOfClassifierChains.class)){
	            				if(((EnsembleOfClassifierChains) mls[i]).isUseClassiferChainUnderSampling()){
	            					s="US"+((EnsembleOfClassifierChains) mls[i]).getNumOfModels();
	            				}
	            			}
	            			if(mls[i].getClass().equals(COCOA.class)){
	            				int numMaxCouples=((COCOA) mls[i]).getNumMaxCouples();
	            				int numCouples=Math.min(mldatas[0].getNumLabels()-1, numMaxCouples);
	            				if(numCouples<=numMaxCouples-10){
	            					System.out.println("!!!!have trained COCOA with numCouples="+numCouples+"!!!!!");
	            					continue;
	            				}
	            				else{
	            					s=numMaxCouples+"_"+numCouples;
	            				}
	            			}
	            			
	            			sb1.append("Original+").append(BaseFunction.Get_Classifier_name(mls[i])+s).append("\n");
		                	//sb2.append("Original+").append(BaseFunction.Get_Classifier_name(mls[i])+s).append("\n");
		                	System.out.print(sb1.toString());
		            	}
	            		else{
	            			String ss="";
	            			
		            		sb1.append(ImUtil.getSamplingName(sp)).append(P).append("+"+BaseFunction.Get_Classifier_name(mls[i])).append("\n");
		            		//sb2.append(ImUtil.getSamplingName(sp)).append(P).append("+"+BaseFunction.Get_Classifier_name(mls[i])).append("\n");
		            		System.out.print(sb1.toString());
		            		sp.setP(P);
	            		}	            		
	    				//BaseFunction.Out_file(outFileImSta+ImUtil.getSamplingName(sp)+P+"ImSta.txt", "", false);
	            		
		            	MultipleEvaluation multiEval=new MultipleEvaluation(mldatas[0]);
		            	ArrayList<ExperimentThread> expThreadList=new ArrayList<>();
		            	CountDownLatch latch=new CountDownLatch(numRepetitions*numFlods);// numRepetitions*numFlods threads co-work
		            	for (int repetition = 0; repetition < numRepetitions; repetition++) {
		    	            // perform 5-fold CV and add each to the current results
		    	            for (int fold = 0; fold < numFlods; fold++) {
		    	            	int index=(repetition * numFlods + fold + 1);
		    	            	System.out.print("Experiment " + index+"\t");
		    	            	//sb2.append("Experiment " + index);
		    	            	
		    	                multiTrain = new MultiLabelInstances(mldatas[repetition].getDataSet().trainCV(numFlods, fold), mldatas[repetition].getLabelsMetaData());
		    	                multiTest = new MultiLabelInstances(mldatas[repetition].getDataSet().testCV(numFlods, fold), mldatas[repetition].getLabelsMetaData());
		    	            	
		    	            	
		    	                if(!sp.getClass().equals(NoProcess.class)){
		    	            		 multiTrain=sp.build(multiTrain);	
				    	             //is.calculateImSta(multiTrain);
				    	        	 //sb2.append("!!!!!!!New Data\n").append(is.toString()).append("\n");
		    	            	}
		    	            	
		    	                ExperimentThread et=new ExperimentThread(mls[i], multiTrain, multiTest,latch);
		    	                et.start();
		    	                expThreadList.add(et);
		    	                
		    	            }
		    	        }
		            	
		            	latch.await();
		            	for(int exIndex=0;exIndex<expThreadList.size();exIndex++){
		            		multiEval.addEvaluation(expThreadList.get(exIndex).getEval());
		            		trainingTime[exIndex]=expThreadList.get(exIndex).getTrainTime();
    	                    testTime[exIndex++]=expThreadList.get(exIndex).getTestTime();
		            	}
		            	
		            	
		            	//BaseFunction.Out_file(outFileImSta+ImUtil.getSamplingName(sp)+P+"ImSta.txt", sb2.toString(), true);
		            	//sb2.delete(0, sb2.length());
		            	
		            	multiEval.calculateStatistics();
		            	for(Measure m:evaluationMeasures){
		            		String measureName=m.getName();		            		
		            		sb1.append(measureName+"\t"+multiEval.getMean(measureName)+"\t"+multiEval.getStd(measureName)+"\n");
		            	}
		            	sb1.append("Training Time\t").append(BaseFunction.Get_Average(trainingTime)+"\t").append(BaseFunction.Get_Std(trainingTime)+"\n");
		            	sb1.append("Test Time\t").append(BaseFunction.Get_Average(testTime)+"\t").append(BaseFunction.Get_Std(testTime)+"\n");
		            	sb1.append("\n");
		            	System.out.println(sb1);
		            	BaseFunction.Out_file(outFileResult, sb1.toString(), true);
		            	sb1.delete(0, sb1.length());	            		
	            	}	
	            }		
			}				
		}

		
		catch(Exception e){
			e.printStackTrace();
		}

	}
	
	
	public static void fTestECCUSVariousModelNumber(){
		//SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
		
		double Ps[]={0.1};
		boolean isRemoveHI=false;
		boolean isHyperion=true;
		
		MultiLabelSampling samplings[]={new NoProcess(),new NoProcess(),new NoProcess(),new NoProcess()}; //,new MutilLabelRandomUnderSampling(),new MutilLabelRandomUnderSampling(),new NoProcess(),new NoProcess(),
		String dataNames[]={"yahoo-Arts1","yahoo-Business1"};//"emotions","yeast","scene","flags","birds","genbase","cal500","enron","medical","mediamill","bibtex"
		
		String  dataName0_1[]={"bibtex","bookmarks","delicious","enron","eurlex-sm","medical"};
		String  dataName0_01[]={"rcv1subset1","rcv1subset2","yahoo-Arts1","yahoo-Business1"};
		HashSet<String> dataName0_1Set=new HashSet<>(); dataName0_1Set.addAll(Arrays.asList(dataName0_1));
		HashSet<String> dataName0_01Set=new HashSet<>(); dataName0_01Set.addAll(Arrays.asList(dataName0_01));
		
		
		Classifier cla=new J48();
		
		EnsembleOfClassifierChains ecc=new EnsembleOfClassifierChains(cla, 10, true, true);
		
		EnsembleBinaryRelevanceUnderSampling ebrus=new EnsembleBinaryRelevanceUnderSampling();
		ebrus.setNumOfModels(10);
		ebrus.setUnderSamplingPercent(1.0);
		ebrus.setUseConfidences(true);
		ebrus.setUseFmeasureOptimizationThreshold(true);	
				
		EnsembleOfClassifierChains eccus=new EnsembleOfClassifierChains(cla, 10, true, true);
		eccus.setUseClassiferChainUnderSampling(true);
		eccus.setUseFmeasureOptimizationThreshold(true);
		eccus.setUnderSamplingPercent(1.0);  //1.0: the equal size of the majority and minority instances after under sampling 

		COCOA cocoa=new COCOA(10);
		cocoa.setUnderSamplingPercent(1.0);
		
		
		int modelNumbers[]={10,20,30,40,50,60,70,80,90,100};

		MultiLabelLearner mls[]={ebrus,ecc,cocoa,eccus};  //
		
        int numRepetitions=2, numFlods=5;   
		long seeds[]=new long[numRepetitions];
		for(int i=0;i<numRepetitions;i++){
			seeds[i]=i+1L;
		}
		
        double trainingTime[]=new double[numRepetitions*numFlods];
        double testTime[]=new double[numRepetitions*numFlods];
		
		try{	
			for(String dataName:dataNames){
				
				System.out.println(dataName);
				

				
				String filePath=isHyperion?"//data//mlkd//BinLiu//MultiLabelDataSet//"+dataName+"//" : "F://刘彬学校电脑资料//希腊//数据//MutliLabel Datasets//"+(isRemoveHI?"RemoveHighImLabel":"")+"//"+dataName+"//";//
				String outFilePath=isHyperion?"//data//mlkd//BinLiu//MultiLabelDataSet//RESULT//22//" : "F://刘彬学校电脑资料//希腊//experiment result//ecc"+(isRemoveHI?"_RemoveHighImLabel":"")+"5"+"//";//  //_RemoveHighImLabel
				String arrfFile=dataName;//+".arff";
				String xmlFile=dataName+".xml";

				if(dataName0_1Set.contains(dataName)){
					arrfFile+="_0.1";
				}
				if(dataName0_01Set.contains(dataName)){
					arrfFile+="_0.01";
				}
				
				System.out.println("Loading the data set");
				MultiLabelInstances []mldatas=new MultiLabelInstances[numRepetitions];
				mldatas[0]=new MultiLabelInstances(filePath+arrfFile+".arff",filePath+xmlFile);
				for(int repetition=1;repetition<numRepetitions;repetition++){
					mldatas[repetition]=mldatas[0].clone();
					mldatas[repetition].getDataSet().randomize(new Random(seeds[repetition]));
				}
				MultiLabelInstances multiTrain=null;
				MultiLabelInstances multiTest=null;
				
				
	            Evaluator evaluator;
	            Measure[] evaluationMeasures = {new MacroFMeasure(mldatas[0].getNumLabels()),new MacroAUC(mldatas[0].getNumLabels()),new MacroAUCPR(mldatas[0].getNumLabels())};
	            	/*
	            	 * ,new MicroFMeasure(mldatas[0].getNumLabels()),new MicroAUC(mldatas[0].getNumLabels()),
	            	 * {new MacroFMeasure(mldata.getNumLabels()),new MacroPrecision(mldata.getNumLabels()),new MacroRecall(mldata.getNumLabels()),
	            								new MacroAUC(mldata.getNumLabels()),new MicroFMeasure(mldata.getNumLabels()),new MicroPrecision(mldata.getNumLabels()),
	            								new MicroRecall(mldata.getNumLabels()),new MicroAUC(mldata.getNumLabels())};
				*/

	            
				String outFileResult=outFilePath+dataName+"Result.txt";
				//String outFileImSta=outFilePath+dataName;  //the imbalanced Statistics of original and processed dataset
				StringBuffer sb1=new StringBuffer();
				//StringBuffer sb2=new StringBuffer();
				BaseFunction.Out_file(outFileResult, "", false);

    	        
				for(int modelNumber:modelNumbers)
        		{
					for(int i=0;i<samplings.length;i++){
						MultiLabelSampling sp=samplings[i];
						double dd[]={0};
	            	
						if(!sp.getClass().equals(NoProcess.class))
							dd=Ps;
	            	
						for(double P:dd){	
		            		if(sp.getClass().equals(NoProcess.class)){
			            		
		            			String s = "";
		            			if(mls[i].getClass().equals(EnsembleOfClassifierChains.class)){
		            				if(((EnsembleOfClassifierChains) mls[i]).isUseClassiferChainUnderSampling()){
		            					mls[i]=new EnsembleOfClassifierChains(cla, modelNumber, true, true);
		            					((EnsembleOfClassifierChains) mls[i]).setUseClassiferChainUnderSampling(true);
		            					((EnsembleOfClassifierChains) mls[i]).setUseFmeasureOptimizationThreshold(true);
		            					((EnsembleOfClassifierChains) mls[i]).setUnderSamplingPercent(1.0);
		            					
		            					s="US"+((EnsembleOfClassifierChains) mls[i]).getNumOfModels();
		            				}
		            				else{
		            					mls[i]=new EnsembleOfClassifierChains(cla, modelNumber, true, true);
		            					s=""+((EnsembleOfClassifierChains) mls[i]).getNumOfModels();
		            				}
		            			}
		            			if(mls[i].getClass().equals(COCOA.class)){
		            				mls[i]=new COCOA(modelNumber);
		            				((COCOA) mls[i]).setUnderSamplingPercent(1.0);
		            				
		            				int numMaxCouples=((COCOA) mls[i]).getNumMaxCouples();
		            				int numCouples=Math.min(mldatas[0].getNumLabels()-1, numMaxCouples);
		            				if(numCouples<=numMaxCouples-10){
		            					System.out.println("!!!!have trained COCOA with numCouples="+numCouples+"!!!!!");
		            					continue;
		            				}
		            				else{
		            					s=numMaxCouples+"_"+numCouples;
		            				}
		            			}
		            			if(mls[i].getClass().equals(EnsembleBinaryRelevanceUnderSampling.class)){
		            				mls[i]=new EnsembleBinaryRelevanceUnderSampling();
		            				((EnsembleBinaryRelevanceUnderSampling)mls[i]).setNumOfModels(modelNumber);
		            				((EnsembleBinaryRelevanceUnderSampling)mls[i]).setUnderSamplingPercent(1.0);
		            				((EnsembleBinaryRelevanceUnderSampling)mls[i]).setUseConfidences(true);
		            				((EnsembleBinaryRelevanceUnderSampling)mls[i]).setUseFmeasureOptimizationThreshold(true);
		            				
		            				s=((EnsembleBinaryRelevanceUnderSampling)mls[i]).getNumOfModels()+"";
		            			}
		            			
		            			sb1.append("Original+").append(BaseFunction.Get_Classifier_name(mls[i])+s).append("\n");
			                	//sb2.append("Original+").append(BaseFunction.Get_Classifier_name(mls[i])+s).append("\n");
			                	System.out.print(sb1.toString());
			            	}
		            		else{
		            			String ss="";
		            			
			            		sb1.append(ImUtil.getSamplingName(sp)).append(P).append("+"+BaseFunction.Get_Classifier_name(mls[i])).append("\n");
			            		//sb2.append(ImUtil.getSamplingName(sp)).append(P).append("+"+BaseFunction.Get_Classifier_name(mls[i])).append("\n");
			            		System.out.print(sb1.toString());
			            		sp.setP(P);
		            		}
		            		
		    				//BaseFunction.Out_file(outFileImSta+ImUtil.getSamplingName(sp)+P+"ImSta.txt", "", false);
		
		            		//System.out.println("P="+P);
		            		
			            	MultipleEvaluation multiEval=new MultipleEvaluation(mldatas[0]);
			            	CountDownLatch latch=new CountDownLatch(numRepetitions*numFlods);
			            	ArrayList<ExperimentThread> expThreadList=new ArrayList<>();
			            	for (int repetition = 0; repetition < numRepetitions; repetition++) {
			    	            // perform 5-fold CV and add each to the current results
			    	            for (int fold = 0; fold < numFlods; fold++) {
			    	            	int index=(repetition * numFlods + fold + 1);
			    	            	System.out.print("Experiment " + index+"\t");
			    	            	//sb2.append("Experiment " + index);
			    	            	
			    	                multiTrain = new MultiLabelInstances(mldatas[repetition].getDataSet().trainCV(numFlods, fold), mldatas[repetition].getLabelsMetaData());
			    	                multiTest = new MultiLabelInstances(mldatas[repetition].getDataSet().testCV(numFlods, fold), mldatas[repetition].getLabelsMetaData());
			    	            	
			    	            	
			    	                if(!sp.getClass().equals(NoProcess.class)){
			    	            		 multiTrain=sp.build(multiTrain);	
					    	             //is.calculateImSta(multiTrain);
					    	        	 //sb2.append("!!!!!!!New Data\n").append(is.toString()).append("\n");
			    	            	}
			    	            	
			    	                ExperimentThread et=new ExperimentThread(mls[i], multiTrain, multiTest,latch);
			    	                et.start();
			    	                expThreadList.add(et);
			    	            }
			    	        }
			            	latch.await();
			            	for(int exIndex=0;exIndex<expThreadList.size();exIndex++){
			            		multiEval.addEvaluation(expThreadList.get(exIndex).getEval());
			            		trainingTime[exIndex]=expThreadList.get(exIndex).getTrainTime();
	    	                    testTime[exIndex++]=expThreadList.get(exIndex).getTestTime();
			            	}
			            	
			            	//BaseFunction.Out_file(outFileImSta+ImUtil.getSamplingName(sp)+P+"ImSta.txt", sb2.toString(), true);
			            	//sb2.delete(0, sb2.length());
			            	
			            	multiEval.calculateStatistics();
			            	for(Measure m:evaluationMeasures){
			            		String measureName=m.getName();		            		
			            		sb1.append(measureName+"\t"+multiEval.getMean(measureName)+"\t"+multiEval.getStd(measureName)+"\n");
			            	}
			            	sb1.append("Training Time\t").append(BaseFunction.Get_Average(trainingTime)+"\t").append(BaseFunction.Get_Std(trainingTime)+"\n");
			            	sb1.append("Test Time\t").append(BaseFunction.Get_Average(testTime)+"\t").append(BaseFunction.Get_Std(testTime)+"\n");
			            	sb1.append("\n");
			            	System.out.println(sb1);
			            	BaseFunction.Out_file(outFileResult, sb1.toString(), true);
			            	sb1.delete(0, sb1.length());		            		
            		
	            		}	
	            	}		
	           }	
	        }			
		}

		
		catch(Exception e){
			e.printStackTrace();
		}
	}

}


class ExperimentThread extends Thread{
	private MultiLabelLearner mlLeaner;
	private MultiLabelInstances mlTrainData;
	private MultiLabelInstances mlTestData;
	private Evaluation eval;
	private double trainTime;
	private double testTime;
	private CountDownLatch latch;
	
	public ExperimentThread(MultiLabelLearner mlLeaner,MultiLabelInstances mlTrainData, MultiLabelInstances mlTestData,CountDownLatch latch ) throws Exception {
		this.mlLeaner=mlLeaner.makeCopy();
		this.mlTrainData=mlTrainData.clone();
		this.mlTestData=mlTestData.clone();	
		this.latch=latch;
	}
	
	
	public double getTrainTime() {
		return trainTime;
	}

	public double getTestTime() {
		return testTime;
	}
	
	public Evaluation getEval(){
		return eval;
	}
	
	public void run() {  
		try{
			SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
			System.out.println(this.getName()+"Start: " + df.format(System.currentTimeMillis()).toString());
			
			
			long beginTime,endTime;
			
			beginTime=System.currentTimeMillis();
            mlLeaner.build(mlTrainData);
            endTime=System.currentTimeMillis();
            trainTime=(endTime-beginTime)/1000.0;
			
			Evaluator evaluator=new Evaluator();
			
			beginTime=System.currentTimeMillis();
			eval = evaluator.evaluate(mlLeaner, mlTestData, mlTrainData);
			endTime=System.currentTimeMillis();
			testTime=(endTime-beginTime)/1000.0;
			
			latch.countDown();
            
		}
		catch (Exception e){
			e.printStackTrace();
		}
		
	}
}