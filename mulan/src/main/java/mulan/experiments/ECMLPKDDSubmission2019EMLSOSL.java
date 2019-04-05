package mulan.experiments;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.ensemble.EnsembleOfSampling;
import mulan.classifier.hypernet.BaseFunction;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.COCOA;
import mulan.classifier.transformation.CalibratedLabelRanking;
import mulan.classifier.transformation.ECCRU23;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.IterativeStratification;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.MacroAUC;
import mulan.evaluation.measure.MacroAUCPR;
import mulan.evaluation.measure.MacroFMeasure;
import mulan.evaluation.measure.Measure;
import mulan.sampling.MLSMOTE;
import mulan.sampling.MLSOL;
import mulan.sampling.MultiLabelSampling;
import mulan.sampling.NoProcess;
import mulan.sampling.REMEDIAL;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * <p>Class replicating the experiment in 
 * <em> Synthetic Oversampling of Multi-Label Data based on Local Label Distribution. ECML-PKDD (2019) [Submission].
 * </em></p>
 *
 * @author Bin Liu
 * @version 2019.3.22
 */

public class ECMLPKDDSubmission2019EMLSOSL {

	public static void main(String[] args) {
		f2();

	}
		
	public static void f2(){
		try{
			boolean isHyperion=false;
			
			Classifier cla =new J48();
			BinaryRelevance br1=new BinaryRelevance(cla);
			MLkNN mlknn=new MLkNN(10, 1.0);
			RAkEL rakel=new RAkEL(new LabelPowerset(cla));
			CalibratedLabelRanking clr=new CalibratedLabelRanking(cla);
			COCOA cocoa=new COCOA(cla,10);
			ECCRU23 eccru3=new ECCRU23(cla,10,true,true); eccru3.setPlusVersion(true);
			MultiLabelLearner baseLearners[]={br1,mlknn,clr,rakel,cocoa,eccru3,}; 
			
			String dataNames[]={
					"flags","genbase","cal500","scene","yeast","enron","medical",
					//"yahoo-Arts1","yahoo-Business1","bibtex","rcv1subset1","rcv1subset2","Corel5k",  
					//other data sets could be obtained from http://mulan.sourceforge.net/datasets-mlc.html
			};		
			String  dataName0_1[]={"bibtex","bookmarks","delicious","enron","eurlex-sm","medical"}; //
			String  dataName0_01[]={"rcv1subset1","rcv1subset2","yahoo-Arts1","yahoo-Business1"};
			HashSet<String> dataName0_1Set=new HashSet<>(); dataName0_1Set.addAll(Arrays.asList(dataName0_1));
			HashSet<String> dataName0_01Set=new HashSet<>(); dataName0_01Set.addAll(Arrays.asList(dataName0_01));

			String fileName="SubmitResults2";
			for(String dataName:dataNames){
				String filePath=isHyperion?"//data//mlkd//BinLiu//MultiLabelDataSet//" : "F://Mulan Codes for Pull//mulan//data//multi-label//";
				String outFilePath=isHyperion?"//data//mlkd//BinLiu//MultiLabelDataSet//RESULT//"+fileName+"//" : "F://刘彬学校电脑资料//希腊//experiment result//ecc"+"5"+"//";
				String arrfFile=filePath+dataName+"//"+dataName+".arff";
				String xmlFile=filePath+dataName+"//"+dataName+".xml";
				
				System.out.println("Loading the data set: "+dataName);
				MultiLabelInstances mldata=new MultiLabelInstances(arrfFile,xmlFile);
				if(dataName0_1Set.contains(dataName)){
					mldata=removeFeatures(mldata,0.1);
				}
				if(dataName0_01Set.contains(dataName)){
					mldata=removeFeatures(mldata,0.01);
				}
				mldata=removeLabelsWith1MinorityInstance(mldata);
				
				MultiLabelInstances multiTrain=null;
				MultiLabelInstances multiTest=null;		
				
				List<Measure> measures = new ArrayList<>(3);
	            measures.add(new MacroFMeasure(mldata.getNumLabels()));
	            measures.add(new MacroAUC(mldata.getNumLabels()));
	            measures.add(new MacroAUCPR(mldata.getNumLabels()));
				
	            System.out.print("Method\t");
	            for(Measure m:measures){
	            	System.out.print(m.getName()+"\t");
	            }
	            System.out.println();
				
				String outFileResult=outFilePath+dataName+"Result.txt";
				BaseFunction.Out_file(outFileResult, "", false);
				
				for(MultiLabelLearner base:baseLearners){
					
					MLSOL mlsosl=new MLSOL(); mlsosl.setP(0.3);
					NoProcess nosampling=new NoProcess();
					MLSMOTE mlsmote=new MLSMOTE();
					REMEDIAL remedial=new REMEDIAL();
			        remedial.setSampling(mlsmote);
					
			        
					EnsembleOfSampling esampling_mlsmote=new EnsembleOfSampling();
					esampling_mlsmote.measure=EnsembleOfSampling.thresholdOptimizationMeasures.Fmeasure;
					esampling_mlsmote.setBaseMlLearner(base); 
					esampling_mlsmote.setMlsampling(mlsmote);  
			        
					EnsembleOfSampling esampling_remedial=new EnsembleOfSampling();
					esampling_remedial.measure=EnsembleOfSampling.thresholdOptimizationMeasures.Fmeasure;
					esampling_remedial.setBaseMlLearner(base); 
					esampling_remedial.setMlsampling(remedial);  
			        
					EnsembleOfSampling esampling_mlsosl=new EnsembleOfSampling();
					esampling_mlsosl.measure=EnsembleOfSampling.thresholdOptimizationMeasures.Fmeasure;
					esampling_mlsosl.setBaseMlLearner(base); 
					esampling_mlsosl.setMlsampling(mlsosl);  
					
					MultiLabelSampling samplings[]={nosampling,mlsmote,remedial,mlsosl,nosampling,nosampling,nosampling};
					MultiLabelLearner learners[]={base,base,base,base,esampling_mlsmote,esampling_remedial,esampling_mlsosl};//,esampling
					
		           

		            for(int i=0;i<Math.min(samplings.length, learners.length);i++){
		            	MultiLabelSampling sp=samplings[i];
		            	MultiLabelLearner ml=learners[i];
		            	MultipleEvaluation results = new MultipleEvaluation(mldata);  
		            	int numRepetitions=5, numFlods=2;
						for (int repetition = 0; repetition < numRepetitions; repetition++) {
		            		IterativeStratification stra=new IterativeStratification(repetition);
		            		MultiLabelInstances mlInss[]=stra.stratify(new MultiLabelInstances(new Instances(mldata.getDataSet()),mldata.getLabelsMetaData()), numFlods);	
		            		for (int fold = 0; fold < numFlods; fold++) {
		    	            	multiTrain=null;
		    	            	for(int j=0;j<numFlods;j++){
		    	            		if(j!=fold){
		    	            			if(multiTrain==null){
		    	            				multiTrain=new MultiLabelInstances(new Instances(mlInss[j].getDataSet()), mlInss[j].getLabelsMetaData());
		    	            			}
		    	            			else{
		    	            				multiTrain.getDataSet().addAll(mlInss[j].getDataSet());
		    	            			}
		    	            		}
		    	            	}
		    	            	multiTest=new MultiLabelInstances(new Instances(mlInss[fold].getDataSet()), mlInss[fold].getLabelsMetaData());
		    	            	
		    	            	Evaluation e1=ftest(sp,ml,multiTrain,multiTest,measures);
		    	            	results.addEvaluation(e1);
		            		}
		    	        }
						StringBuffer sb=new StringBuffer();
						results.calculateStatistics();
						if(! sp.getClass().equals(NoProcess.class)){
							sb.append(BaseFunction.Get_Sampling_name(sp)+"_");
						}
						sb.append(BaseFunction.Get_Classifier_name(ml));
						if(ml.getClass().equals(EnsembleOfSampling.class)){
							sb.append("_"+BaseFunction.Get_Sampling_name(((EnsembleOfSampling)ml).getMlsampling())+"_"+BaseFunction.Get_Classifier_name(((EnsembleOfSampling)ml).getBaseLearner()));
						}
						sb.append("\t");
						for(Measure m:measures){
							sb.append(BaseFunction.Round(results.getMean(m.getName()), 4)+"\t");
						}
						sb.append("\n");
						BaseFunction.Out_file(outFileResult, sb.toString(), true);
						System.out.print(sb.toString());
						
					}
		        }    
			} 
		}
		catch (Exception e){
			e.printStackTrace();
		}
	}

	//remove features with fewer non-zero values
	public static MultiLabelInstances removeFeatures(MultiLabelInstances mlData,double percentRetainFeatures) throws Exception{
		int numFeatures=mlData.getFeatureIndices().length;
		int numRetainFeatures=(int)(numFeatures*percentRetainFeatures);
		int retainFeatureIndices[]=obtainFetainedfeatureIndices(mlData,numRetainFeatures);
		
		
		int[] allRetainAttributes = new int[numRetainFeatures + mlData.getNumLabels()];
        System.arraycopy(retainFeatureIndices, 0, allRetainAttributes, 0, numRetainFeatures);
        int[] labelIndices = mlData.getLabelIndices();
        System.arraycopy(labelIndices, 0, allRetainAttributes, numRetainFeatures, mlData.getNumLabels());

        Remove filterRemove = new Remove();
        filterRemove.setAttributeIndicesArray(allRetainAttributes);
        filterRemove.setInvertSelection(true);
        filterRemove.setInputFormat(mlData.getDataSet());
        Instances filtered = Filter.useFilter(mlData.getDataSet(), filterRemove);
        MultiLabelInstances mlFiltered = new MultiLabelInstances(filtered, mlData.getLabelsMetaData());
        
        return mlFiltered;
	}
	
	protected static int[] obtainFetainedfeatureIndices(MultiLabelInstances mlData,int numRetainFeatures){
		int featureIndices[]=mlData.getFeatureIndices();
		int numFeatures=featureIndices.length;
		int retainFeatureIndices[]=new int[numRetainFeatures];
		
		Instances ins=mlData.getDataSet();
		
		int cs[]=new int[numFeatures];
		Arrays.fill(cs, 0);
		for(Instance data:ins){
			for(int i=0;i<numFeatures;i++){
				int index=featureIndices[i];
				if(data.attribute(index).isNominal()){
					if(!data.stringValue(index).equals("0")){
						cs[i]++;
					}
				}
				else if(data.attribute(index).isNumeric()){
					if(data.value(index)!=0.0D){
						cs[i]++;
					}
				}
				else{
					System.out.println("Can not deal with the not nominal or numeric tyep attribute (attribute index: "+index+")");
				}
			}
		}
		
		HashMap<Integer,ArrayList<Integer>> map=new HashMap<>(); //<number of none zero values, list of feature index>
		for(int i=0;i<numFeatures;i++){	
			if(!map.containsKey(cs[i])){
				ArrayList<Integer> list=new ArrayList<>();
				list.add(featureIndices[i]);
				map.put(cs[i],list);
			}
			else{
				ArrayList<Integer> list=map.get(cs[i]);
				list.add(featureIndices[i]);
				map.put(cs[i],list);
			}
		}
		
		//Sort the map according the number of none zero values
		List<Map.Entry<Integer,ArrayList<Integer>>> list = new ArrayList<Map.Entry<Integer,ArrayList<Integer>>>(map.entrySet());
        Collections.sort(list,new Comparator<Map.Entry<Integer,ArrayList<Integer>>>() {
            //Descending Order
            public int compare(Entry<Integer,ArrayList<Integer>> o1,
                    Entry<Integer,ArrayList<Integer>> o2) {
                return -1*o1.getKey().compareTo(o2.getKey());
            }   
        });
        int c=0;
        boolean isBreak=false;
        for (Iterator iter = list.iterator(); iter.hasNext();){  
        	Map.Entry entry = (Map.Entry)iter.next();  
        	//System.out.println(entry.getKey()+"\t"+Arrays.asList(((ArrayList<Integer>)entry.getValue())).toString());
        	
        	for(int index:((ArrayList<Integer>)entry.getValue())){
        		retainFeatureIndices[c++]=index;
        		if(c>=numRetainFeatures){
        			isBreak=true;
        			break;
        		}
        	}
        	if(isBreak){
        		break;
        	}
        	
        }
		
		return retainFeatureIndices;
	}
	
	//remove labels with only 1 minority instance
	protected static MultiLabelInstances removeLabelsWith1MinorityInstance(MultiLabelInstances mlData) throws Exception{
		Instances ins= mlData.getDataSet();
		int labelIndeices[]=mlData.getLabelIndices();
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
			
			
			if(c1<=1){
				labelIndexToRemove.add(labelIndeices[j]);
			}
		}
		
		MultiLabelInstances mlData2=mlData;
		if(labelIndexToRemove.size()>0){
			int[] labelIndexToRemoveArray=new int[labelIndexToRemove.size()];
			for(int i=0;i<labelIndexToRemove.size();i++){
				labelIndexToRemoveArray[i]=labelIndexToRemove.get(i);
			}
					
			Remove remove = new Remove();
	        remove.setAttributeIndicesArray(labelIndexToRemoveArray);
	        remove.setInputFormat(ins);
	        remove.setInvertSelection(false);
	        
			Instances result=Filter.useFilter(ins, remove);
			mlData2=mlData.reintegrateModifiedDataSet(result);
		}
		return mlData2;

	}
	
	public static Evaluation ftest(MultiLabelSampling mls, MultiLabelLearner mll, MultiLabelInstances mlTrain,MultiLabelInstances mlTest,List<Measure> measures) throws InvalidDataException, Exception{
		 MultiLabelSampling mlsCopy=mls.makeCopy();
		 MultiLabelLearner mllCopy=mll.makeCopy();
		 MultiLabelInstances mlTrainCopy=mlTrain.clone();
		 MultiLabelInstances mlTestCopy=mlTest.clone();	
		 
		 mllCopy.build(mlsCopy.build(mlTrainCopy));
		 
		 Evaluator evaluator = new Evaluator(); 
		 Evaluation ev = evaluator.evaluate(mllCopy, mlTestCopy, measures);
		 return ev;
	}

}
