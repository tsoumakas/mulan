package mulan.sampling;

import java.io.File;
import java.io.FileNotFoundException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Scanner;


import mulan.classifier.MultiLabelLearner;
import mulan.classifier.hypernet.BaseFunction;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.BinaryRelevanceUnderSampling;
import mulan.classifier.transformation.COCOA;
import mulan.classifier.transformation.EnsembleBinaryRelevanceUnderSampling;
import mulan.classifier.transformation.EnsembleOfClassifierChains;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.dimensionalityReduction.SimpleMostFrenquencyReduction;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.*;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibLINEAR;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class SamplingTest {
	private static int seed=1;
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		
		
		String dataNames[]={"yahoo-Arts1","yahoo-Business1","yahoo-Education1"};  
		/*"corel16k" ,"bibtex","birds","bookmarks","cal500","corel5k",
		"delicious","emotions","enron","genbase","medical","scene","tmc2007-500",
		"yeast","eurlex-sm","flags","IMDB-F","mediamill","rcv1subset1"
		,"rcv1subset2"*/
		String filePath="F://刘彬学校电脑资料//希腊//数据//MutliLabel Datasets//";
		
		try{
			
			//simpleFeatureReductionTest();
			
			
			//fTestECCUSVariousModelNumber();
			//fTestECCUS();
			
			//fProcessExperimentResult();
			//fProcessExperiment2Result2();
			//fTest();
			
			//partitionInstances(dataNames);
			//fTestBRUS();
			
			
			ImbalancedStatistics is=new ImbalancedStatistics();
			for(String s:dataNames){
				//System.out.println(s);
				String fileArff=filePath+s+"//"+s+".arff";
				String fileXml=filePath+s+"//"+s+".xml";
				MultiLabelInstances mldata=new MultiLabelInstances(fileArff, fileXml);
				
				
				
				System.out.println("#instances\t"+mldata.getNumInstances());
				System.out.println("#labels\t"+mldata.getNumLabels());
				System.out.println("#features\t"+mldata.getFeatureIndices().length);
				System.out.println("#Cardinality\t"+mldata.getCardinality());
				
				is.calculateImSta(mldata);
				System.out.println(s+"\t"+is.getMeanIRLb()+"\t"+is.getMaxIRLb()+"\t"+is.getCVIR()+"\t"+
						is.getMeanImR()+"\t"+is.getMaxImR()+"\t"+is.getMinImR()+"\t"+is.getSCUMBLE());
				
				
				
				//removeLabelsWithHighImR(mldata, 50.0, filePath+"RemoveHighImLabel//"+s+"//"+s+".arff", filePath+"RemoveHighImLabel//"+s+"//"+s+".xml");
			}
			
		}
		catch(Exception e){
			e.printStackTrace();
		}
		
		
		
	}
	
	

	/**
	 * @param mldata
	 * @param ImRthreshold
	 * @param outFileName
	 * @throws Exception 
	 */
	public static void removeLabelsWithHighImR(MultiLabelInstances mldata,double ImRthreshold,String outFileArff,String outFileXml) throws Exception{
		Instances ins= mldata.getDataSet();
		int labelIndeices[]=mldata.getLabelIndices();
		//String labelNames[]=mldata.getLabelNames();
		//boolean islabelRemove[]=new boolean[labelIndeices.length];
		ArrayList<Integer> labelIndexToRemove=new ArrayList<Integer>();
		
		
		for(int j=0;j<labelIndeices.length;j++){
			int c0=0,c1=0;
			for(Instance data:ins){
				if(data.stringValue(labelIndeices[j]).equals("0")){
					c0++;
				}
				else if(data.stringValue(labelIndeices[j]).equals("1"))
				{	
					c1++;
				}
			}
			
			double ImR=c0*1.0/c1;
			if(ImR<1.0D){
				ImR=c1*1.0/c0;
			}
			System.out.println(j+1+"\t"+ImR+"\t"+c0+"\t"+c1+"\t");
			
			if(ImR>=ImRthreshold||c1<20){
				labelIndexToRemove.add(labelIndeices[j]);
				//islabelRemove[j]=true;
			}
		}
		int[] labelIndexToRemoveArray=new int[labelIndexToRemove.size()];
		for(int i=0;i<labelIndexToRemove.size();i++){
			labelIndexToRemoveArray[i]=labelIndexToRemove.get(i);
		}
		
		/*
		ArrayList<String> labelNamesRetained=new ArrayList<String>();
		for(int j=0;j<islabelRemove.length;j++){
			if(!islabelRemove[j])
				labelNamesRetained.add(labelNames[j]);
		}
		*/		
		
		
		Remove remove = new Remove();
        remove.setAttributeIndicesArray(labelIndexToRemoveArray);
        remove.setInputFormat(ins);
        remove.setInvertSelection(false);
        
		Instances result=Filter.useFilter(ins, remove);
		MultiLabelInstances mldata2=mldata.reintegrateModifiedDataSet(result);
		
		String xmlFileContent=labelNamesToxmlString(mldata2.getLabelNames());
		BaseFunction.Outfile_instances_arff(mldata2.getDataSet(), outFileArff);
		BaseFunction.Out_file(outFileXml, xmlFileContent, false);
		
		System.out.println(mldata2.getNumInstances()+"\t"+mldata2.getNumLabels());
		
		
	}
	
	
	static String labelNamesToxmlString(String labelNames[]){
		StringBuffer sb=new StringBuffer();
		
		sb.append("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<labels xmlns=\"http://mulan.sourceforge.net/labels\">\n");
		for(String s:labelNames){
				sb.append("<label name=\""+s+"\"></label>\n");
		}
		sb.append("</labels>");
					
		return sb.toString();
		
	}
	
	
	
	public static void fTestBRUS() throws Exception{
		String dataName="cal500";
		String filePath="F://刘彬学校电脑资料//希腊//数据//MutliLabel Datasets//RemoveHighImLabel//"+dataName+"//";
		String outFilePath="F://刘彬学校电脑资料//希腊//experiment result//ecc4_RemoveHighImLabel//";
		String arrfFile=dataName;//+".arff";
		String xmlFile=dataName+".xml"; 
				
		BinaryRelevance br=new BinaryRelevance(new J48());
		BinaryRelevanceUnderSampling brus=new BinaryRelevanceUnderSampling(new J48());
		EnsembleOfClassifierChains ecc=new EnsembleOfClassifierChains();
        EnsembleOfClassifierChains eccus=new EnsembleOfClassifierChains();
        
		MultiLabelInstances mldata=new MultiLabelInstances(filePath+arrfFile+".arff",filePath+xmlFile);
		mldata.getDataSet().randomize(new Random(1));
		
		Evaluator evaluator=new Evaluator();
        Measure[] evaluationMeasures = {new MacroFMeasure(mldata.getNumLabels()),new MacroPrecision(mldata.getNumLabels()),new MacroRecall(mldata.getNumLabels()),
        								new MacroSpecificity(mldata.getNumLabels()),new MacroAUC(mldata.getNumLabels()),new MacroAUCPR(mldata.getNumLabels())};
		
        MultipleEvaluation []multiEval=new MultipleEvaluation[4];
        for(int i=0;i<multiEval.length;i++){
        	multiEval[i]=new MultipleEvaluation(mldata);
        }
        
        
        
        int numRepetitions=1, numFlods=5;            
        for (int repetition = 0; repetition < numRepetitions; repetition++) {
            for (int fold = 0; fold < 1; fold++) {
            	int index=(repetition * numFlods + fold + 1);
            	
            	String outFile=outFilePath+dataName+" Detail Analysis of Each label CV"+index +".txt";
                StringBuffer sb=new StringBuffer();
                BaseFunction.Out_file(outFile, "", false);
            	
                
            	MultiLabelInstances multiTrain=new MultiLabelInstances(mldata.getDataSet().trainCV(numFlods, fold),mldata.getLabelsMetaData());
            	MultiLabelInstances multiTest=new MultiLabelInstances(mldata.getDataSet().testCV(numFlods, fold),mldata.getLabelsMetaData());
            	
            	
            	ArrayList <Evaluation> eList=new ArrayList<Evaluation>(3);
            	
            	System.out.println("Training"+BaseFunction.Get_Classifier_name(br));
            	br.build(multiTrain);
            	System.out.println("Testing"+BaseFunction.Get_Classifier_name(br));
                Evaluation e0 = evaluator.evaluate(br, multiTest, multiTrain);
                eList.add(e0);
                multiEval[0].addEvaluation(e0);
                System.out.println("Testing"+BaseFunction.Get_Classifier_name(br));
                
                System.out.println("Training"+BaseFunction.Get_Classifier_name(brus));
                brus.setUnderSamplingPercent(1.0);
                brus.build(multiTrain);
                System.out.println("Testing"+BaseFunction.Get_Classifier_name(brus));
                Evaluation e1 = evaluator.evaluate(brus, multiTest, multiTrain);
                eList.add(e1);
                multiEval[1].addEvaluation(e1);
                
                System.out.println("Training"+BaseFunction.Get_Classifier_name(ecc));
                ecc.build(multiTrain);
                System.out.println("Testing"+BaseFunction.Get_Classifier_name(ecc));
                Evaluation e2 = evaluator.evaluate(ecc, multiTest, multiTrain);
                eList.add(e2);
                multiEval[2].addEvaluation(e2);                
                
                

                eccus.setUseClassiferChainUnderSampling(true);
                eccus.setUseFmeasureOptimizationThreshold(true);
                eccus.setUnderSamplingPercent(1.0);
                System.out.println("Training"+BaseFunction.Get_Classifier_name(eccus)+"_US");
                //MultiLabelInstances multiTrain2=mlus.build(multiTrain);
                eccus.build(multiTrain);
                System.out.println("Testing"+BaseFunction.Get_Classifier_name(eccus)+"_US");
                Evaluation e3 = evaluator.evaluate(eccus, multiTest, multiTrain);
                eList.add(e3);
                multiEval[3].addEvaluation(e3);
                
                /*
                MultiLabelUnderSamplingBasedImRs mlus=new MultiLabelUnderSamplingBasedImRs();
                mlus.setP(0.3);
                System.out.println("Training"+BaseFunction.Get_Classifier_name(ecc)+" and Sampling"+ImUtil.getSamplingName(mlus));
                MultiLabelInstances multiTrain2=mlus.build(multiTrain);
                ecc.build(multiTrain2);
                System.out.println("Testing"+BaseFunction.Get_Classifier_name(br));
                Evaluation e3 = evaluator.evaluate(ecc, multiTest, multiTrain);
                eList.add(e3);
                multiEval[3].addEvaluation(e3);
                */
                
                ImbalancedStatistics ims=new ImbalancedStatistics();
                ims.calculateImSta(multiTrain);
                System.out.println("Orinigal Training set\n"+ims.toString());
                //ims.calculateImSta(multiTrain2);
                //System.out.println("\nSampling Training set\n"+ims.toString());
                
                sb.append("CV"+index+"\n");
            	sb.append("label\tc0\tc1\tImR\tTP\tFP\tFN\tTN\tPrecision\tRecall\tFMeasure\tSpecificity\tAUC\tAUCPR\t\t")
            	.append("TP\tFP\tFN\tTN\tPrecision\tRecall\tFMeasure\tSpecificity\tAUC\tAUCPR\t\t")
            	.append("TP\tFP\tFN\tTN\tPrecision\tRecall\tFMeasure\tSpecificity\tAUC\tAUCPR\t\t")
            	.append("TP\tFP\tFN\tTN\tPrecision\tRecall\tFMeasure\tSpecificity\tAUC\tAUCPR\n");
            	System.out.println(sb.toString());
            	
            	//BaseFunction.Out_file(outFile, sb.toString(), true);

            	for(int i=0;i<mldata.getNumLabels();i++) {	
            		for(int j=0;j<eList.size();j++){
            			Evaluation e=eList.get(j);
                		List<Measure> mlist=e.getMeasures();
                		
                		MacroPrecision mp=(MacroPrecision) mlist.get(11);
                		MacroRecall mr=(MacroRecall) mlist.get(12);
                		MacroFMeasure mf=(MacroFMeasure) mlist.get(13);
                		MacroSpecificity ms=(MacroSpecificity) mlist.get(14);
                		MacroAUC mauc=(MacroAUC) mlist.get(26);
                		MacroAUCPR maucpr=(MacroAUCPR) mlist.get(28);
                		
                		double tps[]=mp.getTruePositives();
                		double fps[]=mp.getFalsePositives();
                		double fns[]=mp.getFalseNegatives();
                		double tns[]=mp.getTrueNegatives();
                		
                		if(j==0){
                    		int c1=(int)tps[i]+(int)fns[i];
                    		int c0=(int)tns[i]+(int)fps[i];
                    		sb.append(i+"\t").append(c0+"\t").append(c1+"\t").append(c0*1.0/c1+"\t");
                		}
                		sb.append(tps[i]).append("\t").append(fps[i]).append("\t").append(fns[i]).append("\t").append(tns[i]).append("\t").
                		append(mp.getValue(i)+"\t").append(mr.getValue(i)+"\t").append(mf.getValue(i)+"\t").append(ms.getValue(i)+"\t")
                		.append(mauc.getValue(i)+"\t").append(maucpr.getValue(i)+"\t\t");
                		if(j==eList.size()-1){
                			sb.append("\n");
                		}
                		
                		if(sb.length()>10000){
                			BaseFunction.Out_file(outFile, sb.toString(), true);
                			sb.delete(0, sb.length());
                		}
                	}
            	}
            	BaseFunction.Out_file(outFile, sb.toString(), true);
    			sb.delete(0, sb.length());
            }
            
        }

	}
	
	
	
	public static void partitionInstances(String [] dataNames) throws InvalidDataFormatException{
		
		for(String dataName:dataNames){
			String filePath="F://刘彬学校电脑资料//希腊//数据//MutliLabel Datasets//"+dataName+"//";
			String outFilePath="F://刘彬学校电脑资料//希腊//experiment result//";
			String arrfFile=dataName;//+".arff";
			String xmlFile=dataName;//+".xml"; 
			int numRepetitions=2, numFlods=5;
			
			
			MultiLabelInstances mldata=new MultiLabelInstances(filePath+arrfFile+".arff", filePath+xmlFile+".xml");
			
			Random rnd=new Random(seed);
			mldata.getDataSet().randomize(rnd);
			for (int repetition = 0; repetition < numRepetitions; repetition++) {
		        for (int fold = 0; fold < numFlods; fold++) {
		        	int index=(repetition * numFlods + fold + 1);
		        	System.out.println("Experiment " + index);
		        	
		            Instances train = mldata.getDataSet().trainCV(numFlods, fold);
		            MultiLabelInstances multiTrain = new MultiLabelInstances(train, mldata.getLabelsMetaData());
		            Instances test = mldata.getDataSet().testCV(numFlods, fold);
		            MultiLabelInstances multiTest = new MultiLabelInstances(test, mldata.getLabelsMetaData());
		            
		            BaseFunction.Outfile_instances_arff(multiTrain.getDataSet(), filePath+arrfFile+"Train-"+index+".arff");
		            BaseFunction.Outfile_instances_arff(multiTest.getDataSet(), filePath+arrfFile+"Test-"+index+".arff");
		        }
			}
		}
	}
	
	public static void fTest(){
		SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
		
		double Ps[]={0.1};
		
		MultiLabelSampling samplings[]={new NoProcess(),new NoProcess(),new NoProcess(),new NoProcess(),new NoProcess(),
				new NoProcess(),new NoProcess(),new NoProcess(),new NoProcess(),new NoProcess(),
				new NoProcess(),new NoProcess(),new NoProcess(),new NoProcess(),new NoProcess(),
				new NoProcess(),new NoProcess(),new NoProcess(),new NoProcess(),new NoProcess()}; //new NoProcess(),new MutilLabelRandomUnderSampling(),new MultiLabelUnderSamplingBasedImRs(),
		ImbalancedStatistics is=new ImbalancedStatistics();
		String dataNames[]={"emotions","scene","birds","flags","genbase","yeast"};//"emotions","yeast","scene","flags","birds","genbase","cal500","enron","medical","mediamill"
		
		EnsembleOfClassifierChains ecc1=new EnsembleOfClassifierChains(new J48(), 10, true, true);
		EnsembleOfClassifierChains ecc3=new EnsembleOfClassifierChains(new J48(), 10, true, true);
		ecc3.setUseClassiferChainUnderSampling(true);
		ecc3.setUseFmeasureOptimizationThreshold(true);
		ecc3.setUnderSamplingPercent(1.0);  //1.0: the equal size of the majority and minority instances after under sampling 
		
		
		
		COCOA cocoa=new COCOA(10);
		cocoa.setUnderSamplingPercent(1.0);
		
		
		ArrayList<MultiLabelLearner> mlList=new ArrayList<>();
		mlList.add(cocoa);
		mlList.add(ecc3);
		for(int i=20;i<=100;i+=10){
			COCOA cocoatemp=new COCOA(i);
			cocoatemp.setUnderSamplingPercent(1.0);
			mlList.add(cocoatemp);
			
			EnsembleOfClassifierChains ecctemp=new EnsembleOfClassifierChains(new J48(), i, true, true);
			ecctemp.setUseClassiferChainUnderSampling(true);
			ecctemp.setUseFmeasureOptimizationThreshold(true);
			ecctemp.setUnderSamplingPercent(1.0); 
			mlList.add(ecctemp);
			
		}
		

		MultiLabelLearner mls[]=new MultiLabelLearner[mlList.size()];
		mls=mlList.toArray(mls); //{cocoa,ecc3};//ecc1,ecc1,ecc1,
		
        int numRepetitions=1, numFlods=5;   
		long seeds[]=new long[numRepetitions];
		for(int i=0;i<numRepetitions;i++){
			seeds[i]=i+1L;
		}
		
		try{	
			for(String dataName:dataNames){
				
				System.out.println(dataName);
				
				boolean isRemoveHI=false;
				String filePath="F://刘彬学校电脑资料//希腊//数据//MutliLabel Datasets//"+(isRemoveHI?"RemoveHighImLabel":"")+"//"+dataName+"//";//
				String outFilePath="F://刘彬学校电脑资料//希腊//experiment result//ecc"+(isRemoveHI?"_RemoveHighImLabel":"")+"3"+"//";  //_RemoveHighImLabel
				String arrfFile=dataName;//+".arff";
				String xmlFile=dataName+".xml";

				
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
	            Measure[] evaluationMeasures = {new MacroFMeasure(mldatas[0].getNumLabels()),new MacroAUC(mldatas[0].getNumLabels()),new MicroFMeasure(mldatas[0].getNumLabels()),new MicroAUC(mldatas[0].getNumLabels()),new MacroAUCPR(mldatas[0].getNumLabels())};
	            	/*{new MacroFMeasure(mldata.getNumLabels()),new MacroPrecision(mldata.getNumLabels()),new MacroRecall(mldata.getNumLabels()),
	            								new MacroAUC(mldata.getNumLabels()),new MicroFMeasure(mldata.getNumLabels()),new MicroPrecision(mldata.getNumLabels()),
	            								new MicroRecall(mldata.getNumLabels()),new MicroAUC(mldata.getNumLabels())};
				*/
	            
         
	            
				String outFileResult=outFilePath+dataName+"Result.txt";
				String outFileImSta=outFilePath+dataName;  //the imbalanced Statistics of original and processed dataset
				StringBuffer sb1=new StringBuffer();
				StringBuffer sb2=new StringBuffer();
				BaseFunction.Out_file(outFileResult, "", false);

				

				/*
				ArrayList <MultiLabelInstances> multiTrainList=new ArrayList <MultiLabelInstances>(numRepetitions*numFlods);
				ArrayList <MultiLabelInstances> multiTestList=new ArrayList <MultiLabelInstances>(numRepetitions*numFlods);
				
				
				for (int repetition = 0; repetition < numRepetitions; repetition++) {
    	            for (int fold = 0; fold < numFlods; fold++) {
    	            	int index=(repetition * numFlods + fold + 1);
    	            	multiTrainList.add(new MultiLabelInstances(filePath+arrfFile+"Train-"+index+".arff",filePath+xmlFile));
    	            	multiTestList.add(new MultiLabelInstances(filePath+arrfFile+"Test-"+index+".arff",filePath+xmlFile));
    	            }
    	        }
    	        */
    	        

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
		                	sb2.append("Original+").append(BaseFunction.Get_Classifier_name(mls[i])+s).append("\n");
		                	System.out.print(sb1.toString());
		            	}
	            		else{
	            			String ss="";
	            			
		            		sb1.append(ImUtil.getSamplingName(sp)).append(P).append("+"+BaseFunction.Get_Classifier_name(mls[i])).append("\n");
		            		sb2.append(ImUtil.getSamplingName(sp)).append(P).append("+"+BaseFunction.Get_Classifier_name(mls[i])).append("\n");
		            		System.out.print(sb1.toString());
		            		sp.setP(P);
	            		}
	            		
	    				BaseFunction.Out_file(outFileImSta+ImUtil.getSamplingName(sp)+P+"ImSta.txt", "", false);
	
	            		//System.out.println("P="+P);
	            		
		            	MultipleEvaluation multiEval=new MultipleEvaluation(mldatas[0]);
		            	for (int repetition = 0; repetition < numRepetitions; repetition++) {
		    	            // perform 5-fold CV and add each to the current results
		    	            for (int fold = 0; fold < numFlods; fold++) {
		    	            	int index=(repetition * numFlods + fold + 1);
		    	            	System.out.print("Experiment " + index+"\t");
		    	            	sb2.append("Experiment " + index);
		    	            	
		    	                multiTrain = new MultiLabelInstances(mldatas[repetition].getDataSet().trainCV(numFlods, fold), mldatas[repetition].getLabelsMetaData());
		    	                multiTest = new MultiLabelInstances(mldatas[repetition].getDataSet().testCV(numFlods, fold), mldatas[repetition].getLabelsMetaData());
		    	            	
		    	            	//multiTrain=multiTrainList.get(index-1);
		    	            	//multiTest=multiTestList.get(index-1);
		    	                
		    	                //is.calculateImSta(multiTrain);
		    	        		//sb2.append("!!!!!!Original Data\n").append(is.toString()).append("\n");
		    	        		
		    	            	if(!sp.getClass().equals(NoProcess.class)){
		    	            		 multiTrain=sp.build(multiTrain);	
				    	             is.calculateImSta(multiTrain);
				    	        	 sb2.append("!!!!!!!New Data\n").append(is.toString()).append("\n");
		    	            	}
		    	            	
		    	            	
		    	            	System.out.print("Begin:"+df.format(System.currentTimeMillis()).toString()+"\t");
		    	            	
		    	                mls[i].build(multiTrain);
		                        evaluator = new Evaluator();
		                        Evaluation e = evaluator.evaluate(mls[i], multiTest, multiTrain);
		                        multiEval.addEvaluation(e);
		                        
		                        System.out.print("End:"+df.format(System.currentTimeMillis()).toString()+"\n");
		    	            }
		    	        }
		            	
		            	BaseFunction.Out_file(outFileImSta+ImUtil.getSamplingName(sp)+P+"ImSta.txt", sb2.toString(), true);
		            	sb2.delete(0, sb2.length());
		            	
		            	multiEval.calculateStatistics();
		            	for(Measure m:evaluationMeasures){
		            		String measureName=m.getName();
		            		System.out.println(measureName+"\t"+multiEval.getMean(measureName)+"\t"+multiEval.getStd(measureName));
		            		sb1.append(measureName+"\t"+multiEval.getMean(measureName)+"\t"+multiEval.getStd(measureName)+"\n");
		            	}
		            	sb1.append("\n");
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
		SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
		
		double Ps[]={0.1};
		
		MultiLabelSampling samplings[]={new NoProcess()}; //,new MutilLabelRandomUnderSampling(),new MutilLabelRandomUnderSampling(),new NoProcess(),new NoProcess(),
		String dataNames[]={"emotions","yeast","scene","flags","birds","genbase","cal500","enron","medical"};//"emotions","yeast","scene","flags","birds","genbase","cal500","enron","medical","mediamill"
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

		MultiLabelLearner mls[]={ebrus};  //ecc,cocoa,eccus
		
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
				
				boolean isRemoveHI=false;
				boolean isHyperion=true;
				
				String filePath=isHyperion?"//data//mlkd//BinLiu//MultiLabelDataSet//"+dataName+"//" : "F://刘彬学校电脑资料//希腊//数据//MutliLabel Datasets//"+(isRemoveHI?"RemoveHighImLabel":"")+"//"+dataName+"//";//
				String outFilePath=isHyperion?"//data//mlkd//BinLiu//MultiLabelDataSet//RESULT//21//" : "F://刘彬学校电脑资料//希腊//experiment result//ecc"+(isRemoveHI?"_RemoveHighImLabel":"")+"5"+"//";//  //_RemoveHighImLabel
				String arrfFile=dataName;//+".arff";
				String xmlFile=dataName+".xml";

				
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
			    	            	
			    	                System.out.print("Begin:"+df.format(System.currentTimeMillis()).toString()+"\t");
			    	            	beginTime=System.currentTimeMillis();
			    	                mls[i].build(multiTrain);
			    	                endTime=System.currentTimeMillis();
			    	                trainingTime[index-1]=(endTime-beginTime)/1000.0;
			    	                
			    	                
			                        evaluator = new Evaluator();
			                       
			                        beginTime=System.currentTimeMillis();
			                        Evaluation e = evaluator.evaluate(mls[i], multiTest, multiTrain);
			                        endTime=System.currentTimeMillis();
			    	                testTime[index-1]=(endTime-beginTime)/1000.0;		    	                
			    	                System.out.print("End:"+df.format(System.currentTimeMillis()).toString()+"\n");
			    	                
			                        multiEval.addEvaluation(e);

			    	            }
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
	
	
	public static void fTestECCUS(){
		SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
		
		double Ps[]={0.1};
		MultiLabelSampling samplings[]={new NoProcess(),new NoProcess(),new NoProcess(),new NoProcess(),
				new MutilLabelRandomUnderSampling(),new MutilLabelRandomUnderSampling(),new NoProcess(),new NoProcess()}; //
		//ImbalancedStatistics is=new ImbalancedStatistics();
		String dataNames[]={"yahoo-Arts1","yahoo-Business1"};//"rcv1subset1","rcv1subset2","bibtex","bookmarks","delicious","eurlex-sm","medical","enron","tmc2007-500",Corel5k
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
		
		

		MultiLabelLearner mls[]={br1,ecc1,brus,ebrus,br1,ecc1,cocoa,eccus}; //br1,ecc1,brus,ebrus,br1,ecc1,cocoa,eccus
		
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
				
				boolean isRemoveHI=false;
				boolean isHyperion=true;
				
				String filePath=isHyperion?"//data//mlkd//BinLiu//MultiLabelDataSet//"+dataName+"//" : "F://刘彬学校电脑资料//希腊//数据//MutliLabel Datasets//"+(isRemoveHI?"RemoveHighImLabel":"")+"//"+dataName+"//";//
				String outFilePath=isHyperion?"//data//mlkd//BinLiu//MultiLabelDataSet//RESULT//3//" : "F://刘彬学校电脑资料//希腊//experiment result//ecc"+(isRemoveHI?"_RemoveHighImLabel":"")+"5"+"//";//  //_RemoveHighImLabel
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
	
	            		//System.out.println("P="+P);
	            		
		            	MultipleEvaluation multiEval=new MultipleEvaluation(mldatas[0]);
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
		    	            	
		    	                System.out.print("Begin:"+df.format(System.currentTimeMillis()).toString()+"\t");
		    	            	beginTime=System.currentTimeMillis();
		    	                mls[i].build(multiTrain);
		    	                endTime=System.currentTimeMillis();
		    	                trainingTime[index-1]=(endTime-beginTime)/1000.0;
		    	                
		    	                
		                        evaluator = new Evaluator();
		                       
		                        beginTime=System.currentTimeMillis();
		                        Evaluation e = evaluator.evaluate(mls[i], multiTest, multiTrain);
		                        endTime=System.currentTimeMillis();
		    	                testTime[index-1]=(endTime-beginTime)/1000.0;		    	                
		    	                System.out.print("End:"+df.format(System.currentTimeMillis()).toString()+"\n");
		    	                
		                        multiEval.addEvaluation(e);

		    	            }
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
	
	
	
	public static void fProcessExperiment2Result2(){
		String filePath="C://Users//lb//Desktop//RESULT//2//";//"F://刘彬学校电脑资料//希腊//experiment result//ecc"+(isRemoveHI?"_RemoveHighImLabel":"")+" USPercent=1.0"+"//";  //_RemoveHighImLabel
		String dataNamesAll[]={"birds","cal500","genbase","medical","scene","yeast","flags"}; //"emotions","enron",
		int numLabels[]={19,174,27,45,6,14,7}; //6,53,
		String measureNames[]={"Macro-averaged F-Measure","Macro-averaged AUC","Macro-averaged AUCPR","Training Time"}; //"Micro-averaged F-Measure","Micro-averaged AUC"
		
		
		for(int i=0;i<dataNamesAll.length;i++){
			String dataNames[]={dataNamesAll[i]};
			ArrayList<ArrayList<String>> methodNameLists=new ArrayList<>();
			
			methodNameLists.add(new ArrayList<String>());
			for(int j=10;j<=100;j+=10){
				methodNameLists.get(methodNameLists.size()-1).add("Original+EnsembleBinaryRelevanceUnderSampling"+j);
			}
			methodNameLists.add(new ArrayList<String>());
			for(int j=10;j<=100;j+=10){
				methodNameLists.get(methodNameLists.size()-1).add("Original+EnsembleOfClassifierChains"+j);
			}
			methodNameLists.add(new ArrayList<String>());
			for(int j=10;j<=100;j+=10){
				if(j>=numLabels[i]){
					methodNameLists.get(methodNameLists.size()-1).add("Original+COCOA"+j+"_"+(numLabels[i]-1));
					break;
				}
				else{
					methodNameLists.get(methodNameLists.size()-1).add("Original+COCOA"+j+"_"+j);
				}
			}
			methodNameLists.add(new ArrayList<String>());
			for(int j=10;j<=100;j+=10){
				methodNameLists.get(methodNameLists.size()-1).add("Original+EnsembleOfClassifierChainsUS"+j);
			}
			
			
			ArrayList<ExperimentResult> list=new ArrayList<ExperimentResult>();
			for(String dataName:dataNames){
				try{
				ArrayList<ExperimentResult> lt=readExperimentResult(filePath+dataName+"Result.txt",dataName);
				list.addAll(lt);
				}
				catch(Exception e){
					e.printStackTrace();
				}
			}

			HashMap<String,Integer> erMap=new HashMap<>();
			for(int j=0;j<list.size();j++){
				ExperimentResult er=list.get(j);
				erMap.put(er.dataName+er.methodName+er.measureName, j);
			}
			
			variousModelNumberResult(measureNames,methodNameLists,dataNames,list,erMap);
			
			/*String methodNames[]={"Original+COCOA10_10","Original+COCOA20_20","Original+COCOA30_30","Original+COCOA40_40","Original+COCOA50_50"
					,"Original+COCOA60_60","Original+COCOA70_70","Original+COCOA80_80","Original+COCOA90_90","Original+COCOA100_100"
					,"Original+EnsembleOfClassifierChainsUS10","Original+EnsembleOfClassifierChainsUS20","Original+EnsembleOfClassifierChainsUS30"
					,"Original+EnsembleOfClassifierChainsUS40","Original+EnsembleOfClassifierChainsUS50","Original+EnsembleOfClassifierChainsUS60"
					,"Original+EnsembleOfClassifierChainsUS70","Original+EnsembleOfClassifierChainsUS80","Original+EnsembleOfClassifierChainsUS90"
					,"Original+EnsembleOfClassifierChainsUS100"};
					//"Original+COCOA30_30","Original+COCOA40_40","Original+COCOA50_44",
			*/
		}
	}
	
	public static void fProcessExperimentResult(){
		boolean isRemoveHI=true;
		String filePath="C://Users//lb//Desktop//RESULT//1//";//"F://刘彬学校电脑资料//希腊//experiment result//ecc"+(isRemoveHI?"_RemoveHighImLabel":"")+" USPercent=1.0"+"//";  //_RemoveHighImLabel
		String dataNames[]={"birds","cal500","genbase","scene","yeast","enron","enron1","medical","medical1"
				,"bibtex","Corel5k","rcv1subset1","rcv1subset2","yahoo-Arts1","yahoo-Business1"}; //"emotions","flags"
		int numLabels[]={19,174,27,6,14,53,53,45,45,159,374,101,101,26,30};
		String measureNames[]={"Macro-averaged F-Measure","Macro-averaged AUC","Macro-averaged AUCPR","Training Time"}; //"Micro-averaged F-Measure","Micro-averaged AUC"
		String methodNames[]={"Original+BinaryRelevance","Original+EnsembleOfClassifierChains","Original+BinaryRelevanceUnderSampling"
				,"Original+EnsembleBinaryRelevanceUnderSampling","MutilLabelRandomUnderSampling0.1+BinaryRelevance","MutilLabelRandomUnderSampling0.1+EnsembleOfClassifierChains"
				,"Original+COCOA","Original+EnsembleOfClassifierChainsUS10"};
		
		ArrayList<ExperimentResult> list=new ArrayList<ExperimentResult>();
		for(String dataName:dataNames){
			try{
			ArrayList<ExperimentResult> lt=readExperimentResult(filePath+dataName+"Result.txt",dataName);
			list.addAll(lt);
			}
			catch(Exception e){
				e.printStackTrace();
			}
		}

		HashMap<String,Integer> erMap=new HashMap<>();
		for(int i=0;i<list.size();i++){
			ExperimentResult er=list.get(i);
			if(er.methodName.indexOf("COCOA")!=-1){
				er.methodName="Original+COCOA";
			}
			erMap.put(er.dataName+er.methodName+er.measureName, i);
		}
		
		performanceComparedResult(measureNames,methodNames,dataNames,list,erMap);
	}
	
	/*
	Measure	Method1				Method2		
	data1	value	std	rank	value	std	rank
	data2						
	data4						
	data3						
	 */		
	public static void performanceComparedResult(String measureNames[],String methodNames[],String dataNames[],ArrayList<ExperimentResult> list,HashMap<String,Integer> erMap ){
		StringBuffer sb=new StringBuffer();
		for(String measureName:measureNames){
			sb.append(measureName+"\t");
			for(String methodName:methodNames){
				sb.append(methodName+"\t\t\t");
			}
			sb.append("\n");
			for(String dataName:dataNames){
				sb.append(dataName+"\t");
				for(String methodName:methodNames){
					Integer i=erMap.get(dataName+methodName+measureName);
					if(i==null){
						sb.append("\t\t\t");
					}
					else{
						ExperimentResult er=list.get(i);
						sb.append(er.value+"\t").append(er.std+"\t").append("\t");
					}
				}
				sb.append("\n");
			}
			System.out.println(sb.append("\n").toString());
			sb.delete(0, sb.length());
		}
	}
	
	
	/*
	Measure	10		20		...	100	
	data1
	Method1 value1	value2  ...	value10
	Method2 value1	value2	...	value10
	data2						
	
	data4						
	
	data3 
	 */
	public static void variousModelNumberResult(String measureNames[],ArrayList<ArrayList<String>> methodNameLists,String dataNames[],ArrayList<ExperimentResult> list,HashMap<String,Integer> erMap ){
		StringBuffer sb=new StringBuffer();		
		for(String measureName:measureNames){
			sb.append(measureName+"\t");
			for(int i=10;i<=100;i+=10){
				sb.append(i+"\t");
			}
			sb.append("\n");
			
			for(String dataName:dataNames){
				sb.append(dataName+"\n");
				
				System.out.println(sb.toString());
				sb.delete(0, sb.length());
				
				for(ArrayList<String> methodNameList:methodNameLists){
					sb.append(methodNameList.get(0).replaceAll("10_6", "").replaceAll("10_5", "").replaceAll("10_10", "").replaceAll("10", "")
							.replaceAll("EnsembleBinaryRelevanceUnderSampling", "EBRUS").replaceAll("EnsembleOfClassifierChains", "ECC").replaceAll("Original\\+","")+"\t");
					int n=10,j=0;
					double lastValue=0.0;
					for(String methodName:methodNameList){
						Integer i=erMap.get(dataName+methodName+measureName);
						if(i==null){
							sb.append("\t");
						}
						else{
							ExperimentResult er=list.get(i);
							lastValue=er.value;
							sb.append(lastValue+"\t");
							
						}
						j++;
					}
					while(j++<n){
						sb.append(lastValue+"\t");
					}
					
					sb.append("\n");
					BaseFunction.Out_file("C:\\Users\\lb\\Desktop\\python code\\plotfiles\\"+measureName+" "+dataName+".txt", sb.toString(), false);
				}
				sb.append("\n");
			}
			System.out.println(sb.append("\n").toString());
			sb.delete(0, sb.length());
		}
	}
	
	
 	public static ArrayList<ExperimentResult> readExperimentResult(String fileName,String dataName) throws FileNotFoundException{
		ArrayList<ExperimentResult> result=new ArrayList<ExperimentResult>();
		
		Scanner in = new Scanner(new File(fileName));
		String methodName="";
		while (in.hasNextLine()) {	
			String str = in.nextLine();
			String sg[] = str.split("\t");
			//System.out.println(sg.length);
			if(sg.length==1 && sg[0].length()>0){
				methodName=sg[0];
			}
			if(sg.length==3){
				result.add(new ExperimentResult(dataName,methodName,sg[0],Double.parseDouble(sg[1]),Double.parseDouble(sg[2])));
			}
		}
		return result;
	}
	
	
	public static void printInformationMLIns(MultiLabelInstances mldata) throws Exception{
		
		System.out.println("# ins:"+mldata.getNumInstances());
		System.out.println("# label:"+mldata.getNumLabels());
		mldata.caculateImbalancedMeasurements();
		System.out.println("# MeanIR:"+mldata.getMeanIR());
		double IRLbl[]=mldata.getIRLbls();
		for(double d:IRLbl){
			System.out.print(d+"\t");
		}
		System.out.println();
		
		Instances ins=mldata.getDataSet();
		int labelIndices[]=mldata.getLabelIndices();
		int c0[]=new int[mldata.getNumLabels()];
		int c1[]=new int[mldata.getNumLabels()];
		for(Instance d:ins){
			for(int i=0;i<mldata.getNumLabels();i++)
			if(d.stringValue(labelIndices[i]).equals("1")){
				c1[i]++;
			}
			else{
				c0[i]++;
			}
		}
		
		for(int i:c1){
			System.out.print(i+"\t");
		}
		System.out.println();
		for(int i:c0){
			System.out.print(i+"\t");
		}
		System.out.println();
	}
	
	
	public static void simpleFeatureReductionTest() throws Exception{
		String dataNames[]={"yahoo-Arts1","yahoo-Business1"};   //"rcv1subset1","rcv1subset2"   "bibtex","bookmarks","delicious","enron","eurlex-sm","medical"
		/*"corel16k" "bibtex","birds","bookmarks","cal500","corel5k",
		"delicious","emotions","enron","genbase","medical","scene","tmc2007-500",
		"yeast","eurlex-sm","flags","IMDB-F","mediamill","rcv1subset1"
		,"rcv1subset2"
		*/
		double percetageRetainedFeatures=0.01;
		
		for(String dataName:dataNames){
			System.out.println(dataName);
			
			String filePath="F://刘彬学校电脑资料//希腊//数据//MutliLabel Datasets//"+dataName+"//";
			
			String arrfFile=dataName;//+".arff";
			String xmlFile=dataName+".xml"; 
			MultiLabelInstances mldata=new MultiLabelInstances(filePath+arrfFile+".arff",filePath+xmlFile);
			SimpleMostFrenquencyReduction fr=new SimpleMostFrenquencyReduction(percetageRetainedFeatures);
			MultiLabelInstances mldata2=fr.build(mldata);
			
			System.out.println("Original #feature\t"+mldata.getFeatureIndices().length);
			System.out.println("Filtered #feature\t"+mldata2.getFeatureIndices().length);
			
			mldata2.getDataSet().setRelationName(dataName+" Retain "+percetageRetainedFeatures+" Features");
			
        	String outFile=filePath+dataName+"_"+percetageRetainedFeatures+".arff";
            BaseFunction.Out_file(outFile, mldata2.getDataSet().toString(), false);
			
		}
		
	}
}


class ExperimentResult{
	public String dataName;
	public String methodName;
	public String measureName;
	public double value;
	public double std;
	
	public ExperimentResult() {
		
	}
	
	public ExperimentResult(String dataName,String methodName, String measureName, double value,double std){
		this.dataName=dataName;
		this.methodName=methodName;
		this.measureName=measureName;
		this.value=value;
		this.std=std;
	}
	
	public String toString(){
		return methodName+"\t"+dataName+"\t"+measureName+"\t"+value+"\t"+std+"\t";
	}
	
	/*
	public boolean equals(Object anObject){
		if (this == anObject) {
            return true;
        }
        if (anObject instanceof ExperimentResult) {
        	ExperimentResult er = (ExperimentResult)anObject;
        	if(this.dataName.equals(er.dataName)&&this.methodName.equals(er.methodName)&&this.measureName.equals(er.measureName)){
				return true;
			}
			else{
				return false;
			}
		}
        return false;
	}
	*/
}
