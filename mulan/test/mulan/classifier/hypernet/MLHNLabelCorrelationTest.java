package mulan.classifier.hypernet;

import org.junit.Before;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelLearnerTestBase;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import weka.classifiers.trees.J48;

public class MLHNLabelCorrelationTest extends MultiLabelLearnerTestBase{


	protected MLHNLabelCorrelation learner;
	
	
	public static void main(String[] args){
		MLHNLabelCorrelationTest test[]=new MLHNLabelCorrelationTest[12];
		for(int i=0;i<test.length;i++){
			test[i]=new MLHNLabelCorrelationTest();
		}
		int i=0;
		try{
			fCV();
			
			//MLkNNTest t=new MLkNNTest();
			//t.setUp(); t.testBuildWith10ClassLabels();
			
			//pass i=0;  System.out.println(i); test[i].setUp(); test[i].testGetTechnicalInformation(); 
			//pass i=1;  System.out.println(i); test[i].setUp(); test[i].testMakeCopy(); 
			//i=2;  System.out.println(i); test[i].setUp(); test[i].testBuild_WithNullDataSet(); 
			//pass i=3;  System.out.println(i); test[i].setUp(); test[i].testBuild_WithMissingValues(); 
			//pass i=4;  System.out.println(i); test[i].setUp(); test[i].testBuildWith10ClassLabels(); 
			//pass i=5;  System.out.println(i); test[i].setUp(); test[i].testSameInputSameOutput(); 
			//pass i=6;  System.out.println(i); test[i].setUp(); test[i].testBuild_WithDifferentOrder(); 
			//pass i=7;  System.out.println(i); test[i].setUp(); test[i].testBuildWithNonSparse(); 
			//pass i=8;  System.out.println(i); test[i].setUp(); test[i].testBuild(); 
			//i=9;  System.out.println(i); test[i].setUp(); test[i].testMakePrediction_WithNullData(); 
			//i=10; System.out.println(i); test[i].setUp(); test[i].testMakePrediction_BeforeBuild();
			//pass i=11; System.out.println(i); test[i].setUp(); test[i].testMakePrediction_Generic();
			
		}
		catch(Exception e){
			e.printStackTrace();
		}
		
	}
	
	static void fCV(){
		String datapath="F://ÑÐ¾¿Éú//git//mulan//data//multi-label//";
		String dataname="emotions"; //emotions,CAL500,enron,yeast
		String arffFilename = datapath + dataname+"//"+dataname +".arff";
		String xmlFilename = datapath + dataname+"//"+dataname + ".xml";
		
		
		
		MLHNLabelCorrelation mlhnglc=new MLHNLabelCorrelation();
		mlhnglc.setSeed(100L);
		MLHNLabelCorrelation mlhnglc2=new MLHNLabelCorrelation();
		mlhnglc.setSeed(586L);
		BinaryRelevance br=new BinaryRelevance(new J48());
		
		MultiLabelLearner mls[]={mlhnglc,mlhnglc,br};
		String mName[]={"Hamming Loss","Subset Accuracy","Example-Based F Measure",
				"Micro-averaged F-Measure","Coverage","Ranking Loss","Average Precision"};
		try{
			
			MultiLabelInstances dataset = new MultiLabelInstances(arffFilename, xmlFilename);

				for(MultiLabelLearner ml:mls){
					//System.out.println(BaseFuction.Get_Classifier_name(ml));
					//System.out.println("Train Finished");
					Evaluator eval = new Evaluator();
					MultipleEvaluation result = eval.crossValidate(ml,dataset,5);
					//System.out.println("Test Finished");
						
					for(String s:mName){
						System.out.println(s+"\t"+result.getMean(s)+"\t"+result.getStd(s));
					}
					System.out.println();
				}
				

		}
		catch(Exception e){
			e.printStackTrace();
		}
	}
	
	
    @Override
    protected MultiLabelLearnerBase getLearner() {
        return learner;
    }

    @Before
    public void setUp() throws Exception{
    	MultiLabelLearner baseLeaner =new BinaryRelevance(new J48());
    	MultiLabelHyperNetWorkType type=MultiLabelHyperNetWorkType.MLHN_GC;
        learner = new MLHNLabelCorrelation(baseLeaner,type);
    }

}
