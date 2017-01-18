package mulan.classifier.hypernet;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import mulan.classifier.InvalidDataException;
import mulan.classifier.ModelInitializationException;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.MultiLabelInstances;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
/**
 * Class of implementation of Multi-Label Hypernetwork for exploiting global and local label correlation
* @author LB
* @version 2017.01.10
*/

public class MLHNLabelCorrelation extends MultiLabelLearnerBase{

	private static final long serialVersionUID = 1L;

	private ArrayList<HyperEdge> E=null;
	
	//parameters
	private MultiLabelLearner baseLeaner=new BinaryRelevance(new J48()); 
	private int EachNum=5;  //the number of hyperedge generated by each instance
	private int learnNumRP=10; //the iteration number of hyperedge replacing
	private int learnNumGD=10; //the iteration number of gradientDescent
	private double intialRepaleceRate=0.4;
	private double learnRate=0.002;
	private double matchThreshold=0.05;
	private int order=3;
	private double alpha=0.5; //base learner Weight
	private MultiLabelHyperNetWorkType type=MultiLabelHyperNetWorkType.MLHN_GLC;
	private boolean isProcessedData=false;  //true if the input data is processed, 
				    //false if the input data is unprocessed and should be learned by base leaner at first 
		
	private Instances processTrainDataset=null;
	//private int numLabels=-1; 
	private double labelUnbalanced[]=null;
	
	private int preTrainLabel[][]=null;
	private double preConfidence[][]=null;
	private ArrayList<ArrayList<Integer>> TrainMatchIndex=null;   
	//record the index of hyperedge than is matched with processed training data
	private double preThreld[]=null;
	
	
	public MLHNLabelCorrelation() {
		
	}
	
	public MLHNLabelCorrelation(MultiLabelLearner baseLeaner,MultiLabelHyperNetWorkType type){
		try{
			this.baseLeaner=baseLeaner.makeCopy();
		}
		catch (Exception e){
			e.printStackTrace();
		}
		finally{
			this.type=type;
		}
		
	}
	public MLHNLabelCorrelation(MultiLabelHyperNetWorkType type,boolean isProcessedData){
		this.type=type;
		this.isProcessedData=isProcessedData;
	}
	
	
	public void setEachNum(int EachNum){
		this.EachNum=EachNum;
	}
	public void setlearnNumRP(int learnNumRP){
		this.learnNumRP=learnNumRP;
	}
	public void setlearnNumGD(int learnNumGD){
		this.learnNumGD=learnNumGD;
	}
	public void setlearnRate(double learnRate){
		this.learnRate=learnRate;
	}
	public void setmatchThreshold(int learnmatchThresholdNumRP){
		this.matchThreshold=learnmatchThresholdNumRP;
	}
	public void setOrder(int order){
		this.order=order;
	}
	public void setType(MultiLabelHyperNetWorkType type){
		this.type=type;
	}
	public void setisProcessedData(boolean b){
		this.isProcessedData=b;
	}
	public void setAlpha(double a){
		this.alpha=a;
	}
	
	public int getEachNum(){
		return this.EachNum;
	}
	public int getlearnNumRP(){
		return this.learnNumRP;
	}
	public int getlearnNumGD(){
		return this.learnNumGD;
	}
	public double getlearnRate(){
		return this.learnRate;
	}
	public double getmatchThreshold(){
		return this.matchThreshold;
	}
	public int getOrder(){
		return order;
	}
	public MultiLabelHyperNetWorkType getType(){
		return this.type;
	}
	public boolean getisProcessedData(){
		return this.isProcessedData;
	}
	public double getAlpha(){
		return this.alpha;
	}
	
	/**
	 * Bulids the processed trianing data set if the isProcessedData is false
	 * @param train the training multi-label instances
	 * @throws InvalidDataException if build processed trianing data set fails
	 * @throws Exception if the training of baseLeaner fails 
	 */
	private void buildProcessTrainDataset(MultiLabelInstances train) throws InvalidDataException, Exception{
		baseLeaner.build(train);
		 
		 ArrayList<Attribute> attriList=new ArrayList<Attribute>();
		 //predicting label
		 for(Attribute attri:train.getLabelAttributes()){
			 Attribute aNumeric=new Attribute(attri.name()+"prelabel");
			 attriList.add(aNumeric);
		 }
		 //predicting confidence
		 for(Attribute attri:train.getLabelAttributes()){
			 Attribute aNumeric=new Attribute(attri.name()+"preconfidence");
			 attriList.add(aNumeric);
		 }
		 //truth label
		 for(Attribute attri:train.getLabelAttributes()){
			 Attribute aNumeric=new Attribute(attri.name());
			 attriList.add(aNumeric);
		 }
		 
		 processTrainDataset=new Instances("TrainDataSetProcessedBy"+BaseFunction.Get_Classifier_name(baseLeaner), 
				 attriList, train.getNumInstances());
		 System.out.println("Buliding processed training data set");
		 
		 Instances trainIns=train.getDataSet();
		 //numLabels=train.getNumLabels(); //Initialize numLabels
		 int labelIndeics[]=train.getLabelIndices();
		 for(int i=0; i<train.getNumInstances();i++){
			 MultiLabelOutput output=baseLeaner.makePrediction(trainIns.get(i));
			 double attriValue[]=new double[3*numLabels];
			 boolean preLabel[]=output.getBipartition();
			 double preConfidence[]=output.getConfidences();
			 //predicting label
			 for(int j=0;j<numLabels;j++){
				 if(preLabel[j])
					 attriValue[j]=1;
				 else
					 attriValue[j]=0;
			 }
			//predicting confidence
			 for(int j=0;j<numLabels;j++){
				 attriValue[numLabels+j]=preConfidence[j];
			 }
			//truth label
			 for(int j=0;j<numLabels;j++){
				 if(Math.abs(trainIns.get(i).value(labelIndeics[j])-1.0)<1e-8)
					 attriValue[2*numLabels+j]=1;
				 else
					 attriValue[2*numLabels+j]=0;
			 }
			 processTrainDataset.add(new DenseInstance(1.0,attriValue));
		 }
	}
	 
	@Override
	protected void buildInternal(MultiLabelInstances train) throws Exception{
		
		if(isProcessedData){
			processTrainDataset=train.getDataSet();
		}
		else{
			buildProcessTrainDataset(train);
		}
		
		initialize();
		caculateLabelUnbalanced();
		hyperEdgeReplace();
		preTrainLabel=new int[processTrainDataset.numInstances()][numLabels];
		preConfidence=new double[processTrainDataset.numInstances()][numLabels];
		initialPreThreld();
		gradientDescent();
	 }
	 
	
	private void initialize() throws Exception{
		this.E=new ArrayList<HyperEdge>();
		//different types need different initialize functions
		switch(type){
		case MLHN_GC: 
			initializeGC(); break;
		case MLHN_LC: 
			initializeLC(); break;
		case MLHN_GLC:
			initializeGLC(); break;
		}
	}
	
 	private void initializeGC() throws Exception{
		 for(int i=0;i<processTrainDataset.numInstances();i++){
			 Instance data=processTrainDataset.get(i);
			 int c=0;
			 while(c<EachNum){
				 Integer vertexArray[]=BaseFunction.randomIntegerArray(numLabels-1,0,order);
				 HyperEdge e=new HyperEdge(data, vertexArray, numLabels,i);
				 E.add(e);
				 c++;
			 }
		 }
	 }
	private void initializeLC() throws Exception{
		 for(int i=0;i<processTrainDataset.numInstances();i++){
			 Instance data=processTrainDataset.get(i);
			 int c=0;
			 while(c<EachNum){
				 Integer vertexArray[]=BaseFunction.randomIntegerArray(2*numLabels-1,numLabels,order);
				 Boolean valueTypeArray[]=new Boolean[order];
				 for(int j=0;j<order;j++){
					 valueTypeArray[j]=false;
				 }
				 c++;
				 E.add(new HyperEdge(data, vertexArray, valueTypeArray,numLabels,i));
			 }
		 }
	 }
	private void initializeGLC() throws Exception{
		 for(int i=0;i<processTrainDataset.numInstances();i++){
			 Instance data=processTrainDataset.get(i);
			 int c=0;
			 while(c<EachNum){
				 Integer vertexArray[]=BaseFunction.randomIntegerArray(numLabels-1,0,order);
				 Boolean valueTypeArray[]=new Boolean[order];
				 for(int j=0;j<order;j++){
					 valueTypeArray[j]=(BaseFunction.randomInt(0, 1)==0);
					 if(!valueTypeArray[j]){
						 vertexArray[j]+=numLabels;
					 }
				 }
				 c++;
				 E.add(new HyperEdge(data, vertexArray, valueTypeArray,numLabels,i));
			 }
		 }
	 }
	 
	
	private void replaceGC(int beginIndex,int endIndex) throws Exception{
		 if(beginIndex<0||beginIndex>E.size()-1||endIndex<0||endIndex>E.size()-1)
			 return;
		 for(int i=beginIndex;i<=endIndex;i++){
			int index=E.get(i).classIndex;
			Instance data=processTrainDataset.get(index);
			Integer vertexArray[]=BaseFunction.randomIntegerArray(numLabels-1,0,order);
			E.set(i, new HyperEdge(data, vertexArray, numLabels,index));
	   	}				
	 }
	private void replaceLC(int beginIndex,int endIndex) throws Exception{
		 if(beginIndex<0||beginIndex>E.size()-1||endIndex<0||endIndex>E.size()-1)
			 return;
		 for(int i=beginIndex;i<=endIndex;i++){
			int index=E.get(i).classIndex;
			Instance data=processTrainDataset.get(index);
			 Integer vertexArray[]=BaseFunction.randomIntegerArray(2*numLabels-1,numLabels,order);
			 Boolean valueTypeArray[]=new Boolean[order];
			 for(int j=0;j<order;j++){
				 valueTypeArray[j]=false;
			 }
			E.set(i, new HyperEdge(data, vertexArray, valueTypeArray,numLabels,index));
	   	}				
	 }
	private void replaceGLC(int beginIndex,int endIndex) throws Exception{
		 if(beginIndex<0||beginIndex>E.size()-1||endIndex<0||endIndex>E.size()-1)
			 return;
		 for(int i=beginIndex;i<=endIndex;i++){
			int index=E.get(i).classIndex;
			Instance data=processTrainDataset.get(index);
			Integer vertexArray[]=BaseFunction.randomIntegerArray(numLabels-1,0,order);
			Boolean valueTypeArray[]=new Boolean[order];
			for(int j=0;j<order;j++){
				valueTypeArray[j]=(BaseFunction.randomInt(0, 1)==0);
				if(!valueTypeArray[j]){
					vertexArray[j]+=numLabels;
				}
			}
			E.set(i, new HyperEdge(data, vertexArray, valueTypeArray,numLabels,index));
	   	}				
	 }
	 	 
	private void caculateLabelUnbalanced(){
		 labelUnbalanced=new double [numLabels];
		 for(int j=0;j<numLabels;j++){
			 labelUnbalanced[j]=0;
		 }
		
		 for(int i=0;i<processTrainDataset.numInstances();i++){
			 Instance data=processTrainDataset.get(i);
			 for(int j=0;j<numLabels;j++){
				 if(data.value(2*numLabels+j)==1.0){
					 labelUnbalanced[j]+=1;
				 }
			 }
		 }
		 
		 for(int j=0;j<numLabels;j++){
			 if(labelUnbalanced[j]!=0.0)
				 labelUnbalanced[j]/=(processTrainDataset.numInstances()-labelUnbalanced[j])/labelUnbalanced[j];
		 }
	 }
	 
	 //startIndex/endIndex is the start/end index of E that need to update fitness value;  
	private void caculateFitness(int startIndex,int endIndex){		
		 int matchNum;
		 double maxFit,aveFit;
		 for(int i=startIndex;i<endIndex;i++){
			matchNum=0;
			int matchRightNum[]=new int[numLabels];
			for(Instance data:processTrainDataset){
				HyperEdge e=E.get(i);
				if(e.isMatch(data, matchThreshold)){
					matchNum++;
					int results[]=E.get(i).classifyInstance(data,numLabels);
					for(int j=0;j<numLabels;j++){
						matchRightNum[j]+=results[j];
					}
				}
				
				maxFit=aveFit=0;
				for(int j=0;j<numLabels;j++){
					double d;
					if(e.getLabel(j)==1&&labelUnbalanced[j]>1.0){
						d=matchRightNum[j]*labelUnbalanced[j]/(matchNum+matchRightNum[j]*(labelUnbalanced[j]-1));
					}
					else{
						d=matchRightNum[j]*1.0/matchNum;
					}
					e.setWeight(d,j);
					if (maxFit<d)
						maxFit=d;
					aveFit+=d;
				}
				aveFit/=numLabels;
				e.setFitness(maxFit*0.5+aveFit*0.5);
				E.set(i, e);
			}

		 }
	 }
	 
	private void hyperEdgeReplace() throws Exception{ 
		double r=intialRepaleceRate;
		caculateFitness(0,E.size()-1);	
		for(int t=0;t<learnNumRP;t++){
			Collections.sort(E,new Comparator<HyperEdge>()
			{   public int compare(HyperEdge arg0, HyperEdge arg1) {
			         return arg0.fitness.compareTo(arg1.fitness)*-1;  //Desc order
				}
			});
			
			int subNum=(int) (E.size()*r);
			r*=0.9;
			switch (type){
			case MLHN_GC: 
				replaceGC(E.size()-subNum, E.size()-1); break;
			case MLHN_LC: 
				replaceLC(E.size()-subNum, E.size()-1); break;
			case MLHN_GLC:
				replaceGLC(E.size()-subNum, E.size()-1); break;		
			}
			caculateFitness(E.size()-subNum,E.size()-1);
		}
		
		

	 }
		
	private void initialTrainMatchIndex(){
		this.TrainMatchIndex=new ArrayList<ArrayList<Integer>>();
		for(Instance data:processTrainDataset){
			 ArrayList<Integer> list=new ArrayList<Integer>();
			 for(int i=0;i<E.size();i++){
				 if(E.get(i).isMatch(data,matchThreshold)){
					 list.add(i);
				 }
			 }
			 TrainMatchIndex.add(list);
		 }
	}
	
	private void initialPreThreld(){
		preThreld=new double[numLabels];
		for(int i=0;i<preThreld.length;i++)
			preThreld[i]=0.5;
	}
	
	private void makePredictionTrain(){
		
		ArrayList<Integer> matchList=null;
		for(int i=0;i<processTrainDataset.size();i++){
			Instance data=processTrainDataset.get(i);
			matchList=TrainMatchIndex.get(i);
			double w1[]=new double[numLabels];
			double w0[]=new double[numLabels];
			
			for(int index:matchList){
				HyperEdge e=E.get(index);
				for(int j=0;j<numLabels;j++){
					if(e.getLabel(j)==1){
						w1[j]+=e.getWeight(j);
					}
					else{
						w0[j]+=e.getWeight(j);
					}
				}
			}
			
			for(int j=0;j<numLabels;j++){
				double d=0;
				if(w1[j]+w0[j]!=0){
					d=w1[j]/(w1[j]+w0[j]);
				}
				
				preConfidence[i][j]=data.value(numLabels+j)*alpha+d*(1-alpha);	
			}
		}
		
		
		for(int i=0;i<processTrainDataset.size();i++){
			for(int j=0;j<numLabels;j++){
				if(preConfidence[i][j]>preThreld[j])
					preTrainLabel[i][j]=1;
			}
		}
	}
	
	private void gradientDescent(){
		initialTrainMatchIndex();
		ArrayList<Integer> matchList=null;
		double changeW[]=new double[numLabels];
		for(int t=0;t<learnNumGD;t++){
			makePredictionTrain();
			for(int i=0;i<processTrainDataset.size();i++){
				 Instance data=processTrainDataset.get(i);
				 matchList=TrainMatchIndex.get(i);
				 
				 ArrayList<Integer> wrongLabelList=new ArrayList<Integer>();
				 for(int j=0;j<numLabels;j++){
					 int truthLabel=(int) data.value(2*numLabels+j);
					 if(truthLabel!=preTrainLabel[i][j]){
						 wrongLabelList.add(j);
					 } 
				 }
				 
				 for(int index:matchList){
					 HyperEdge e=E.get(index);
					 for(int j:wrongLabelList){
						int truthLabel=(int) data.value(2*numLabels+j);
						if(truthLabel==e.getLabel(j))
						{
							if(truthLabel==1){
								 e.updateWeigth(learnRate*(1-preConfidence[i][j]), j);
							 }
							 else{
								 e.updateWeigth(learnRate*(1-(1-preConfidence[i][j])), j);
							 }			 
						 }
					 }
					 E.set(index, e);
				 }
			 }
		 }
		makePredictionTrain();  //caculate preThreld
	 }
	

	 

	private MultiLabelOutput makePredictionbyMLHN(Instance processedTest){
		boolean[] bipartition = new boolean[numLabels];
	    double[] confidences = new double[numLabels];
	    
	    double w0[]=new double[numLabels];
	    double w1[]=new double[numLabels];
	    for(HyperEdge e:E){
	    	if(e.isMatch(processedTest, matchThreshold)){
	    		for(int j=0;j<numLabels;j++){
	    			if(e.getLabel(j)==1){
	    				w1[j]+=e.getWeight(j);
	    			}
	    			else{
	    				w0[j]+=e.getWeight(j);
	    			}
	    		}
	    	}
	    }
	    
	    for(int j=0;j<numLabels;j++){
			double d=0;
			if(w1[j]+w0[j]!=0){
				d=w1[j]/(w1[j]+w0[j]);
			}
			
			confidences[j]=processedTest.value(numLabels+j)*alpha+d*(1-alpha);
			
			if(confidences[j]>preThreld[j]){
				bipartition[j]=true;
			}
		}
	    
	    
	    MultiLabelOutput result = new MultiLabelOutput(bipartition, confidences);
	    return result;
	}
	 
	@Override
	protected MultiLabelOutput makePredictionInternal(Instance test) throws InvalidDataException, ModelInitializationException, Exception {
		Instance processedTest=null; 
	    if(isProcessedData){
	    	processedTest=test;
	    }
	    else{
			MultiLabelOutput baseOut=baseLeaner.makePrediction(test);
		    double baseConfidence[]=baseOut.getConfidences();
		    boolean baseBipartition[]=baseOut.getBipartition();
		    
		    int featureNum=test.numAttributes()-numLabels;
			processedTest=new DenseInstance(processTrainDataset.get(0));
			
			for(int j=0;j<numLabels;j++){
				 if(baseBipartition[j])
					 processedTest.setValue(j, 1);
				 else
					 processedTest.setValue(j, 0);
			 }
			//predicting confidence
			 for(int j=0;j<numLabels;j++){
				 processedTest.setValue(numLabels+j, baseConfidence[j]);
			 }
			//truth label
			 for(int j=0;j<numLabels;j++){
				 if(Math.abs(test.value(featureNum+j)-1.0)<1e-8)
					 processedTest.setValue(2*numLabels+j, 1);
				 else
					 processedTest.setValue(2*numLabels+j, 1);
			 } 
	    }

	     
	     MultiLabelOutput finalout = makePredictionbyMLHN(processedTest);
	     return finalout;
	 }

	@Override
	public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.INCOLLECTION);
        result.setValue(Field.AUTHOR, "Bin, Liu");
        result.setValue(Field.TITLE, "Multi-Label Hypernetwork for exploiting global and local label correlation");
       // result.setValue(Field.PAGES, "");
       // result.setValue(Field.BOOKTITLE, "");
       // result.setValue(Field.EDITOR, "");
       // result.setValue(Field.PUBLISHER, "");
       // result.setValue(Field.EDITION, "");
        result.setValue(Field.YEAR, "2016");
        return result;
	}
	
	
}
	


