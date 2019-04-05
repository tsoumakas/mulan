package mulan.sampling;

import java.util.Arrays;

import mulan.data.ImbalancedStatistics;
import mulan.data.MultiLabelInstances;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/***
 * <p>Implementation of the REMEDIAL.</p> <p>For
 * more information, see <em> Charte, Francisco, et al.
 *  "Dealing with difficult minority labels in imbalanced mutilabel data sets." 
 *  Neurocomputing 326-327 (2019): 39-53.</em></p>
 *
 * @author Bin Liu
 * @version 2019.3.19
 * 
 */

public class REMEDIAL extends MultiLabelSampling {
	protected MultiLabelSampling sampling;  //use REMEDIAL as the pre-process of a multi-label sampling method if sampling!=null 
	
	/**
	 * @return the sampling
	 */
	public MultiLabelSampling getSampling() {
		return sampling;
	}


	/**
	 * @param sampling the sampling to set
	 */
	public void setSampling(MultiLabelSampling sampling) {
		this.sampling = sampling;
	}


	@Override
	public MultiLabelInstances build(MultiLabelInstances mlDataset) throws Exception {
		ImbalancedStatistics is=new ImbalancedStatistics();
		is.calculateImSta(mlDataset);
		double scumble=is.getSCUMBLE();
		double scumbleIns[]=is.getSCUMBLEs();
		double IRLbls[]=is.getIRLbls();
		double IRLbl=is.getMeanIR();
		
		MultiLabelInstances newMlDataset=null;
		Instances ins=mlDataset.getDataSet();
		int numInsOri=ins.numInstances();
		int labelIndices[]=mlDataset.getLabelIndices();
		int numLabels=mlDataset.getNumLabels();
			
		Instances ins2=new Instances(ins);
		for(int i=0;i<numInsOri;i++){
			if(scumbleIns[i]>scumble){
				Instance data=ins2.get(i);  //Maintain majority labels
				Instance dataCopy=new DenseInstance(data);  //Maintain minority labels
				dataCopy.setDataset(ins2);
				for(int j=0;j<numLabels;j++){
					if(IRLbls[j]<=IRLbl){  //minority label
						data.setValue(labelIndices[j], "0"); 
					}
					else{  //majority instance
						dataCopy.setValue(labelIndices[j], "0");
					}
				}
				ins2.add(dataCopy);
			}
		}
		
		newMlDataset=new MultiLabelInstances(ins2, mlDataset.getLabelsMetaData());
		if(sampling!=null){
			sampling.setSeed(seed);
			newMlDataset=sampling.build(newMlDataset);
		}
		return newMlDataset;
	}
	
	
	

}
