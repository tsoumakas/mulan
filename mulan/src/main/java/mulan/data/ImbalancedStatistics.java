package mulan.data;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;

/***
 * Calculation of measurements for imbalance level of multi-label dataset 
 * 
 * @author Bin Liu
 * @version 2018.5.1
 *
 */

public class ImbalancedStatistics implements Serializable, TechnicalInformationHandler {

	
	
	private static final long serialVersionUID = 1206845794397561644L; 
	
	private int c1[];  //number of instances associated with each label
	private int c0[];  //number of instances not associated with each label
    private double IRLbls[]=null;  //IRLbl of each label
    private double ImRs[]=null;  //ImR of each label
    private double meanIR;  //mean value of IRLbls
    private double meanImR; //mean value of ImRs

	private double CVIR;
	private double CVImR;
	private double maxIRLb;
    private double maxImR;
    private double minIRLb;
   	private double minImR;
    private double SCUMBLE;
    private double SCUMBLEs[];
        
    private int maxC1;

	private double npl;// the percetange of labels that do not contain positive instance
    

	/** labelsets and their frequency */
    private HashMap<LabelSet, Integer> labelsetMap; 
    
    
    /**
	 * @return the npl
	 */
	public double getNpl() {
		return npl;
	}

    
    /**
	 * @return the c1
	 */
	public int[] getC1() {
		return c1;
	}


	/**
	 * @return the c0
	 */
	public int[] getC0() {
		return c0;
	}


	/**
	 * @return the iRLbls
	 */
	public double[] getIRLbls() {
		return IRLbls;
	}
	
	
	public double getIRLblj(int j){
		if(j<0 || j>IRLbls.length){
			System.out.println("!!Invalid Index!!");
			return -1;
		}
		return IRLbls[j];
	}
	
	public double getImRj(int j){
		if(j<0 || j>=ImRs.length){
			System.out.println("!!Invalid Index!!");
			return -1;
		}
		return ImRs[j];
	}

	/**
	 * @return the imRs
	 */
	public double[] getImRs() {
		return ImRs;
	}

	public double getMeanImR() {
		return meanImR;
	}
		
	public double getMeanIR() {
		return meanIR;
	}


	/**
	 * @return the cVIR
	 */
	public double getCVIR() {
		return CVIR;
	}
    
	/**
	 * @return the cVImR
	 */
	public double getCVImR() {
		return CVImR;
	}

	/**
	 * @return the maxIR
	 */
	public double getMaxIRLb() {
		return maxIRLb;
	}


	/**
	 * @return the maxImR
	 */
	public double getMaxImR() {
		return maxImR;
	}
	
	 /**
		 * @return the minIRLb
	 */
	public double getMinIRLb() {
		return minIRLb;
	}


	/**
	 * @return the minImR
	 */
	public double getMinImR() {
		return minImR;
	}	
	
	
	/**
	 * @return the SCUMBLE
	 */
	public double getSCUMBLE() {
		return SCUMBLE;
	}


	/**
	 * @return the SCUMBLEs
	 */
	public double[] getSCUMBLEs() {
		return SCUMBLEs;
	}	
	

	
	/**
	 * @return the maxC1
	 */
	public int getMaxC1() {
		return maxC1;
	}


	public void calculateC0C1(MultiLabelInstances mlData){
		int numInstances = mlData.getNumInstances();
        int numLabels = mlData.getNumLabels();
        int[] labelIndices = mlData.getLabelIndices();
        Instances dataSet=mlData.getDataSet();
        
		c1=new int[numLabels];
		c0=new int[numLabels];
		 //calculate c1, c0
        for (int i = 0; i < numInstances; i++) {
        	Instance ins=dataSet.get(i);
        	for (int j = 0; j < numLabels; j++) {    
        		/* Returns the value of a nominal, string, date, 
        		  or relational attribute for the instance as a string. */
        		if (ins.stringValue(labelIndices[j]).equals("1")) {
                    c1[j]++;
                }
        		else if(ins.stringValue(labelIndices[j]).equals("0")){
        			c0[j]++;
        		}
            }
        }  
        maxC1=Arrays.stream(c1).max().getAsInt();
	}
	
	public void calculateC0C1ImRs(MultiLabelInstances mlData){
		calculateC0C1(mlData);
		int numLabels = mlData.getNumLabels();
		ImRs = new double[numLabels];
		int min=0,max=0;
        for (int j = 0; j < numLabels; j++) {
        	min=c1[j]<c0[j]?c1[j]:c0[j];
        	max=c1[j]>c0[j]?c1[j]:c0[j];
    		ImRs[j]=max*1.0/min;
        }
	}
	
	/** 
     * calculates various multi-label imbalanced statistics, such as ImR,
     * IRLbl, CVIR and SCUMBLE
     * 
     * @param mlData a multi-label dataset
     */
	public void calculateImSta(MultiLabelInstances mlData){
    	int numInstances = mlData.getNumInstances();
        int numLabels = mlData.getNumLabels();
        int[] labelIndices = mlData.getLabelIndices();
        Instances dataSet=mlData.getDataSet();
        
        c1=new int[numLabels];
        c0=new int[numLabels];
        IRLbls = new double[numLabels];
    	ImRs = new double[numLabels];
        SCUMBLEs=new double[numInstances];
       
        
        meanIR=0;
        meanImR=0;
        CVIR=0;
        CVImR=0;
        SCUMBLE=0;
        
        //calculate c1, c0
        for (int i = 0; i < numInstances; i++) {
        	Instance ins=dataSet.get(i);
        	for (int j = 0; j < numLabels; j++) {    
        		/* Returns the value of a nominal, string, date, 
        		  or relational attribute for the instance as a string. */
        		if (ins.stringValue(labelIndices[j]).equals("1")) {
                    c1[j]++;
                }
        		else if(ins.stringValue(labelIndices[j]).equals("0")){
        			c0[j]++;
        		}
            }
        }        
        maxC1=Arrays.stream(c1).max().getAsInt();
        
        //calculate IRLbls and ImRs
        int mc=0;
        for (int j = 0; j < numLabels; j++) {
        	if(c1[j]>mc){
        		mc=c1[j];
        	}
        }
        int min=0,max=0;
        for (int j = 0; j < numLabels; j++) {
        	min=c1[j]<c0[j]?c1[j]:c0[j];
        	max=c1[j]>c0[j]?c1[j]:c0[j];
        	IRLbls[j]=mc*1.0/c1[j];
    		ImRs[j]=max*1.0/min;
        }
        
        
        //calculate MeanIR, MaxIRLb, MaxImR, MinIRLb, MinImR
        maxIRLb=Double.MIN_VALUE;   maxImR=Double.MIN_VALUE;	
        minIRLb=Double.MAX_VALUE;   minImR=Double.MAX_VALUE;	
        int cIR=0,cImR=0;
        for (int j = 0; j < numLabels; j++) {
        	if(Double.isFinite(IRLbls[j])){  //IRLbls[j]=NaN when c1[j]=0
        		meanIR+=IRLbls[j];
        		if(maxIRLb<IRLbls[j]){
        			maxIRLb=IRLbls[j];
        		}
        		if(minIRLb>IRLbls[j]){
        			minIRLb=IRLbls[j];
        		}
        		cIR++;
        	}
        	
        	if(Double.isFinite(ImRs[j])){  //ImRs[j]=NaN when min(c0[j],c1[j])=0
        		meanImR+=ImRs[j];
        		if(maxImR<ImRs[j]){
        			maxImR=ImRs[j];
        		}
        		if(minImR>ImRs[j]){
        			minImR=ImRs[j];
        		}
        		cImR++;
        	}
        	
        }
        meanIR/=cIR;
        meanImR/=cImR;
        
        //calculate CVIR, CVImR
        for(double d:IRLbls){
        	if(Double.isFinite(d) && d<Double.MAX_VALUE)
        		CVIR+=Math.pow((d-meanIR), 2.0);
    	}
        CVIR=Math.sqrt(CVIR/(cIR-1))/meanIR;
        for(double d:ImRs){
        	if(Double.isFinite(d) && d<Double.MAX_VALUE)
        		CVImR+=Math.pow((d-meanImR), 2.0);
        }
        CVImR=Math.sqrt(CVImR/(cImR-1))/meanImR;
        
        //calculate SCUMBLE, SCUMBLEs
        SCUMBLE=0;
        for(int i=0;i<numInstances;i++){
        	Instance ins=dataSet.get(i);
        	double pro=1,ave=0;
        	int c=0;
        	for(int j=0;j<numLabels;j++){
        		if(ins.stringValue(labelIndices[j]).equals("1")){
        			pro*=IRLbls[j];  //Do not considering Double.MAX_VALUE
        			ave+=IRLbls[j];
        			c++;
        		}
        	}
        	if(c==0){	
        		SCUMBLEs[i]=0.0;
        	}
        	else{
            	ave/=c;
            	SCUMBLEs[i]=1-(1/ave)*Math.pow(pro, 1.0/c);
        	}

        	SCUMBLE+=SCUMBLEs[i];
        }
        SCUMBLE/=numInstances;
        
        
        
        //calculate npl
        npl=0;
        for(int c:c1){
        	if(c==0){
        		npl++;
        	}
        }
        npl/=numLabels;
        
        
        
        
    }
    
	
	@Override
 	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("meanIR: ").append(meanIR).append("\n");
		sb.append("CVIR: ").append(CVIR).append("\n");
		sb.append("maxIRLb: ").append(maxIRLb).append("\n");
		sb.append("minIRLb: ").append(minIRLb).append("\n");
		sb.append("meanImR: ").append(meanImR).append("\n");
		sb.append("maxImR: ").append(maxImR).append("\n");
		sb.append("minImR: ").append(minImR).append("\n");
		sb.append("SCUMBLE: ").append(SCUMBLE).append("\n");
		return sb.toString();
	}
	
	public String toStringDetails(){
		StringBuilder sb = new StringBuilder();
		sb.append("meanIR: ").append(meanIR).append("\n");
		sb.append("CVIR: ").append(CVIR).append("\n");
		sb.append("maxIRLb: ").append(maxIRLb).append("\n");
		sb.append("minIRLb: ").append(minIRLb).append("\n");
		sb.append("meanImR: ").append(meanImR).append("\n");
		sb.append("maxImR: ").append(maxImR).append("\n");
		sb.append("minImR: ").append(minImR).append("\n");
		sb.append("SCUMBLE: ").append(SCUMBLE).append("\n");
		 
		sb.append("Measurements:\t").append("c1\t").append("c0\t").append("IRLbls\t").append("ImRs").append("\n");
	    for (int j = 0; j < c1.length; j++) {
	    	sb.append("label ").append(j + 1).append(": \t").append(c1[j]+"\t").append(c0[j]+"\t").append(IRLbls[j]+"\t").append(ImRs[j]+"\t").append("\n");
	    }
	     
		return sb.toString();
	}


	@Override
	public TechnicalInformation getTechnicalInformation() {
		// TODO Auto-generated method stub
		return null;
	}

}
