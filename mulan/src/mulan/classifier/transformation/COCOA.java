package mulan.classifier.transformation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

import mulan.classifier.InvalidDataException;
import mulan.classifier.ModelInitializationException;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.measure.InformationRetrievalMeasures;
import mulan.transformations.COCOATripleClassTransformation;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.supervised.instance.SpreadSubsampleWithMissClassValues;


/**
 * <p>Implementation of the  Cross-Coupling Aggregation (COCOA) algorithm.</p> <p>For
 * more information, see <em> Zhang, Min-Ling, Yu-Kun Li, and Xu-Ying Liu. 
 * "Towards Class-Imbalance Aware Multi-Label Learning." IJCAI. 2015.</em></p>
 *
 * @author Bin Liu
 * @version 2017.12.19
 */


public class COCOA extends TransformationBasedMultiLabelLearner{

	protected BinaryRelevanceUnderSampling brus;
	protected Classifier[][] triClassifiers;
	protected COCOATripleClassTransformation trt;
    protected int[][] triLabelIndices;
    int numCouples;  //the number of coupling class labels
    int numMaxCouples=10;
	protected double thresholds[];   
	/**
	 * Percentage of majority class instances for each label to be deleted
	 * The 1.0 value denotes that the equal size of the majority and minority instances after under sampling 
	 */
    protected double underSamplingPercent=1.0;
	protected int seed=1;
    


	public COCOA(){
    	super();
    }
    
    public COCOA(int numMaxCouples){
    	this();
    	this.numMaxCouples=numMaxCouples;
    }
    
    public COCOA(Classifier baseClassifier,int numMaxCouples){
    	this.baseClassifier=baseClassifier;
    	this.numMaxCouples=numMaxCouples;
    	
    }
    
	public int getNumCouples() {
		return numCouples;
	}
	
	public int getNumMaxCouples() {
		return numMaxCouples;
	}
    

	public double getUnderSamplingPercent() {
		return underSamplingPercent;
	}

	public void setUnderSamplingPercent(double underSamplingPercent) {
		this.underSamplingPercent = underSamplingPercent;
	}
		
	public int getSeed() {
		return seed;
	}
			
	public void setSeed(int seed) {
		this.seed = seed;
	}	

	
	
	//select numCouples label indices from list
    protected int[] selectedLabelIndices(ArrayList<Integer> labelIndexList,int currentLabelIndex){
    	int result[]=new int[numCouples];
    	for(int i_list=0,i_array=0;i_array<numCouples;i_list++){
    		int l=labelIndexList.get(i_list);
    		if(l!=currentLabelIndex){
    			result[i_array++]=l;
    		}
    	}
    	
    	return result;
    }
    
    //random under sampling of the triple class instances 
    protected Instances TrirandomUnderSampling(Instances ins) throws Exception{
    	Instances result = null;
    	int numClass=ins.numClasses();
    	int c[]=new int[numClass];
    	Arrays.fill(c, 0);
    	for(Instance data:ins){
    		if(data.classIsMissing()){
    			continue;
    		}
    		int i=Integer.parseInt(data.stringValue(ins.classIndex()));
    		c[i]++;
    	}
    	int maxi=0,midi=1,mini=2;
    	for(int i=0;i<numClass;i++){
    		if(c[maxi]<c[i]){
    			 maxi=i;
    		}
    		if(c[mini]>c[i]){
    		     mini=i;
    		}
    	}
    	for(int i=0;i<numClass;i++){
    		if(i!=maxi&&i!=mini){
    			midi=i;
    		}
    	}
    	
    	double d1=c[maxi]*(1.0-underSamplingPercent)/c[mini],d2=c[midi]*(1.0-underSamplingPercent)/c[mini];
    	if(d1<1.0 && d2<1.0){
    		result=new Instances(ins);
    		
    		SpreadSubsampleWithMissClassValues ss=new SpreadSubsampleWithMissClassValues();
    		ss.setInputFormat(ins);
    		ss.setRandomSeed(seed);
    		ss.setDistributionSpread(1.0);
    		result=ss.useFilter(ins, ss);
    		
    	}
    	else{
    		result=new Instances(ins,0);
    		
    		{
    			Instances result1=new Instances(ins);
        		SpreadSubsampleWithMissClassValues ss=new SpreadSubsampleWithMissClassValues();
        		ss.setInputFormat(ins);
        		ss.setRandomSeed(seed);
        		ss.setDistributionSpread(d1);  //d1>=1.0
        		result1=ss.useFilter(result1, ss);
        		
        		for(Instance data:result1){
        			if((int)data.classValue()!=midi){
        				result.add(data);
        			}
        		}
    		}
    		
    		{
    			Instances result2=new Instances(ins);
    			SpreadSubsampleWithMissClassValues ss=new SpreadSubsampleWithMissClassValues();
        		ss.setInputFormat(ins);
        		ss.setRandomSeed(seed);
        		if(d2<=1.0){
        			ss.setDistributionSpread(1.0);
        		}
        		else{
        			ss.setDistributionSpread(d2);  //d2>=1.0
        		}
        		result2=ss.useFilter(result2, ss);
        		
        		for(Instance data:result2){
        			if((int)data.classValue()==midi){
        				result.add(data);
        			}
        		}
    		}
    	}
    	
    	return result;
    }
    
    //calculate the thresholds to distinguish the relative and irrelative instances of each label based on optimizing the Macro F-measure
    protected void calculateThresholds(MultiLabelInstances trainingSet) throws Exception{    	
    	double predictConfidences[][]=new double [trainingSet.getNumInstances()][trainingSet.getNumLabels()];
   	    
    	for(int i=0;i<trainingSet.getNumInstances();i++){
    		Instance data=trainingSet.getDataSet().get(i);
    		predictConfidences[i]=Arrays.copyOf(this.makePredictionforThreshold(data),numLabels); 
    	}
    	
    	for(int j=0;j<numLabels;j++){	
    		double maxF=Double.MIN_VALUE;
    		boolean truelabels[]=new boolean[trainingSet.getNumInstances()];
    		for(int i=0;i<trainingSet.getNumInstances();i++){
        		truelabels[i]=trainingSet.getDataSet().get(i).stringValue(labelIndices[j]).equals("1");
        	}
        	for(double d=0.05D;d<0.951D;d+=0.05D){
        		int tp=0,fp=0,fn=0;
        		for(int i=0;i<trainingSet.getNumInstances();i++){
        			boolean preidctLabel=predictConfidences[i][j]>=d;
        			if(preidctLabel){
        				if(truelabels[i])
        					tp++;
        				else
        					fp++;
        			}
        			else{
        				if(truelabels[i]){
        					fn++;
        				}
        			}
        		}				 
        		double f=InformationRetrievalMeasures.fMeasure(tp,fp,fn,1.0);
        		if(f>maxF){
        			maxF=f;
        			thresholds[j]=d;
        		}
        		
        	}
    	}
    }
    
    
    
	@Override
	protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
		numCouples=Math.min(numLabels-1, numMaxCouples);
		brus=new BinaryRelevanceUnderSampling(this.baseClassifier,underSamplingPercent,seed);
		
		trt=new COCOATripleClassTransformation(trainingSet);
		triLabelIndices=new int[numLabels][];
		triClassifiers=new Classifier[numLabels][];
		for(int j=0;j<numLabels;j++){
			triLabelIndices[j]=new int[numCouples];
			triClassifiers[j]=new Classifier[numCouples];
			for(int k=0;k<numCouples;k++){
				triClassifiers[j][k]=AbstractClassifier.makeCopy(baseClassifier);
			}
		}		
		thresholds=new double[numLabels];
		
		
		brus.build(trainingSet);  //training binary classifiers with under sampling training sets
		
		ArrayList<Integer> labelIndexList=new ArrayList<Integer>(numLabels);
		for(int i:labelIndices){
			labelIndexList.add(i);
		}
		Random rnd=new Random(seed);
		
		
		for(int j=0;j<numLabels;j++){
			
			Collections.shuffle(labelIndexList, rnd);
			triLabelIndices[j]=selectedLabelIndices(labelIndexList,labelIndices[j]);
			for(int k=0;k<numCouples;k++){
				
				Instances triClassIns=trt.transformInstances(labelIndices[j], triLabelIndices[j][k]);
				Instances usTriClassIns=TrirandomUnderSampling(triClassIns);
				triClassifiers[j][k].buildClassifier(usTriClassIns);
			    
			}						
		}
		calculateThresholds(trainingSet);
		
	}
	
	//only use triClassifiers to predict training set or not
	private  double[] makePredictionforThreshold(Instance instance) throws InvalidDataException, ModelInitializationException, Exception{	
		double[] confidences = new double[numLabels];
		Arrays.fill(confidences,0);
		
		MultiLabelOutput outBrus=brus.makePrediction(instance);
		for(int j=0;j<numLabels;j++){
			confidences[j]+=outBrus.getConfidences()[j];
		}
				
		for(int j=0;j<numLabels;j++){
			for(int k=0;k<numCouples;k++){
				Instance triData=trt.transformInstance(instance, labelIndices[j], triLabelIndices[j][k]);
				double d[]=triClassifiers[j][k].distributionForInstance(triData);
				confidences[j]+=d[2];
			}
			confidences[j]/=(numCouples+1);
		}
		
		return confidences;
	}
	
	@Override
	protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception, InvalidDataException {
		double confidences[]=makePredictionforThreshold(instance);
		boolean bipartition[]=new boolean[numLabels];
				
		
		for(int j=0;j<numLabels;j++){
			if(confidences[j]>thresholds[j]){
				bipartition[j]=true;
			}
			else{
				bipartition[j]=false;
			}
		}
		
		return new MultiLabelOutput(bipartition, confidences);
	}
	
	

}
