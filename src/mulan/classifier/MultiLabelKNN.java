package mulan.classifier;

import weka.classifiers.Classifier;
import weka.filters.*;
import weka.filters.unsupervised.attribute.Remove;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.neighboursearch.CoverTree;


/**
 * Class that implements the ML-KNN (Multi-Label K Nearest Neighbours ) algorithm <p>
 *
 * @author Eleftherios Spyromitros Xioufis
 */

public class MultiLabelKNN extends AbstractMultiLabelClassifier{

	private double [] PriorProbabilities;
	private double [] PriorNProbabilities;
	private double [][] CondProbabilities;
	private double [][] CondNProbabilities;

	//private int numLabels;
	private int numofNeighbours;
	private double smooth;

	public MultiLabelKNN(){}

	public MultiLabelKNN(int labels,int k,double s)
	{
		numLabels = labels;
		numofNeighbours = k;
		smooth = s ;
		PriorProbabilities = new double [numLabels];
		PriorNProbabilities = new double [numLabels];
		CondProbabilities = new double [numLabels][numofNeighbours+1];
		CondNProbabilities = new double [numLabels][numofNeighbours+1];
	}

	public void buildClassifier(Instances traindata) throws Exception{
		//filtrarisma ton dedomenon kai perasma sti sinartisi
		//ComputePrior gia ypologismo ton Prior Probabilities
		
		Instances datalabels = transform(traindata,false);
		
		ComputePrior(datalabels);
		ComputeCond (traindata);
		
		

	}

	/**
	 * Ypologizei tis prior probabilities tis kathe klasis dexete os eisodo ta
	 * dedomena pou periexoun mono ta class attributes kai me vasi ayta
	 * ypologizei to periexomeno ton pinakon PriorProbabilities kai
	 * PriorNProbabilities
	 * 
	 * @param filteredata
	 */
	private void ComputePrior(Instances datalabels){
		double [] values; // piankas pou apothikeuei tis times gia kathe
							// attribute, olon ton instances
		int temp_Ci;
		for(int i=0;i<datalabels.numAttributes();i++) {// i<numLabels
				values = datalabels.attributeToDoubleArray(i);
				temp_Ci=0;
				for(int j=0 ;j< datalabels.numInstances();j++) // compute the sum
					temp_Ci += (int) values[j];
				PriorProbabilities[i]=  (smooth +temp_Ci)/(smooth*2+datalabels.numInstances());
				PriorNProbabilities[i] = 1 - PriorProbabilities[i];
		}
	}
	
	private void ComputeCond (Instances train){
		//-1 einai o class index
		//diladi den exei oristei clasi
		//kai to covertree douleui giati vriskei  tinapostasi me vasi ta
		//attributes apo 0 mexrin numattributes kai aplos prosperna tin
		//klasi. alla afou einai -1 den ti vriskei pote
		//System.out.println(datalabels.classIndex());
		boolean ncomputed = false;
		Instances attributes = null;
		Instances temp = null;
		
		CoverTree myCoverTree = null;
		try {
			attributes = transform(train,true);
			
			myCoverTree = new CoverTree();
			
			myCoverTree.setInstances(attributes);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

			

		for(int i=0;i<1;i++){
			for(int j=0;j<train.numInstances();j++){
				try {
					temp = myCoverTree.kNearestNeighbours(attributes
							.instance(j), numofNeighbours);
				} catch (Exception e) {
					// TODO: handle exception
				}
				
				if(temp.numInstances()!= 2)

				System.out.println(temp.numInstances());
			}
		}
	}
	
	
	
	/**
	 * Remove all non - label attributes or all label attributes depending on
	 * parameter
	 * 
	 * @param option
	 * if True select attributes
	 * if False select labels
	 * @param train
	 */
	private Instances transform(Instances train,boolean option) throws Exception
	{
		// Indices of attributes to keep or to remove
		int predictors = train.numAttributes()- numLabels;
		int indices[] = new int[predictors];

		
		for (int j = 0; j < predictors; j++){
				indices[j] = j;
			}

		Remove remove = new Remove();
		remove.setInvertSelection(option);
		remove.setAttributeIndicesArray(indices);
		remove.setInputFormat(train);
		Instances result = Filter.useFilter(train, remove);
		return result;
	}




}
