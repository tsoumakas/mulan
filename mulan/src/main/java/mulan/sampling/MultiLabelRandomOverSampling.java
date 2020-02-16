package mulan.sampling;

import java.util.ArrayList;
import java.util.Random;

import mulan.data.ImbalancedStatistics;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;

/***
* <p>Implementation of the MLROS.</p> <p>For
* more information, see <em> Charte, Francisco, et al. 
* "Addressing imbalance in multilabel classification: Measures and random resampling algorithms."
*  Neurocomputing 163 (2015): 3¨C16.</em></p>
*
* @author Bin Liu
 */

public class MultiLabelRandomOverSampling extends MultiLabelSampling{

	public MultiLabelInstances build(MultiLabelInstances mlDataset) throws Exception{
		Random rnd=new Random(seed);
		
		MultiLabelInstances newMlDataset=null;
		Instances ins=mlDataset.getDataSet();
		
		int oriInsNum=mlDataset.getNumInstances();
		int addInsNum=(int)(P*oriInsNum);
		int labelNum=mlDataset.getNumLabels();
		int labelIndices[]=mlDataset.getLabelIndices();
		boolean isMinLabel[]=new boolean[labelNum];
		ArrayList<Integer> addInsIndexList=new ArrayList<Integer>();
		
		ImbalancedStatistics is=new ImbalancedStatistics();
		is.calculateImSta(mlDataset);
		
		double IRLbls[]=is.getIRLbls();
		double MeanIR=is.getMeanIR();
		int c1[]=is.getC1();
		
		for(int i=0;i<labelNum;i++){
			if(IRLbls[i]>MeanIR){
				isMinLabel[i]=true;
			}
		}
		
		ArrayList<ArrayList<Integer>> minBags=new ArrayList<ArrayList<Integer>>();
		for(int i=0;i<labelNum;i++)
		{
			minBags.add(new ArrayList<Integer>());
		}
		
		
		
		for(int i=0;i<ins.numInstances();i++){
			Instance d=ins.get(i);
			for(int j=0;j<labelNum;j++){
				//d.attribute(labelIndices[j]).value((int)d.value(labelIndices[j]));
				if(d.stringValue(labelIndices[j]).equals("1")) 
				{
					if(isMinLabel[j])
						minBags.get(j).add(i);
				}
			}
		}
		int mc=Integer.MIN_VALUE;
		for(int i:c1){
			if(mc<i)
				mc=i;
		}
		
		boolean isAllMajority;
		while(addInsNum>0){
			
			isAllMajority=true;
			for(boolean b:isMinLabel){
				if(b){
					isAllMajority=false;
				}
			}
			if(isAllMajority){
				break;  
				/*
				 *  when all labels is majority (<= MeanIR) there is another solution that recalculating MeanIR, IRLbl, c1[] and mc
				 */
			}
			
			
			//Iterator <ArrayList<Integer>>it= majBags.iterator();
	        for(int i=0; i<labelNum; i++){
	        	if(!isMinLabel[i])
	        		continue;  //not a minority label
	        	
	        	//recalculate IRLbl of label i
	        	double newIRLbl= mc*1.0/c1[i]; 
	            if(newIRLbl<=MeanIR){
	            	isMinLabel[i]=false;  //tag label i to the majority label
	            	continue;
	            }
	               
	            int k=ImUtil.randomInt(0, minBags.get(i).size()-1,rnd);
	            int insIndex=minBags.get(i).get(k);
	            
	            for(int j=0;j<labelNum;j++){
	            	if(ins.get(insIndex).stringValue(labelIndices[j]).equals("1")){
	            		c1[j]++;
					}
	            }	      
	            addInsIndexList.add(insIndex);
	            addInsNum--;
	            
	            if(addInsNum<=0){
	            	break;
	            }
	        }
		}
		
		
		Instances ins2=new Instances(ins);
		for(int i:addInsIndexList){
			ins2.add(ins.get(i));
		}
		newMlDataset=new MultiLabelInstances(ins2, mlDataset.getLabelsMetaData());
		
		return newMlDataset;
	}
	

}
