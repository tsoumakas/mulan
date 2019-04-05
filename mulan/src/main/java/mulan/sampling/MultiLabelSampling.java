package mulan.sampling;

import java.io.Serializable;
import mulan.data.MultiLabelInstances;
import weka.core.SerializedObject;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;

/**
 * Superclass of all multi-label sampling algorithms
 *
 * @author Bin Liu
 */

public abstract class MultiLabelSampling implements TechnicalInformationHandler, Serializable {
	
	protected double P=0.1; //Percentage of instances to be deleted or generated
	
	protected int seed=0;
	
	/**
	 * @return the p
	 */
	public double getP() {
		return P;
	}


	/**
	 * @param p the p to set
	 */
	public void setP(double p) {
		P = p;
	}


	/**
	 * @param seed the seed to set
	 */
	public void setSeed(int seed) {
		this.seed = seed;
	}

	
	
	public abstract MultiLabelInstances build(MultiLabelInstances mlDataset) throws Exception;
	
	
	/**
     * Creates a deep copy of the given sampling method using serialization.
     *
     * @return a deep copy of the sampling method
     * @exception Exception if an error occurs while making copy of the sampling method.
     */
    public MultiLabelSampling makeCopy() throws Exception {
        return (MultiLabelSampling) new SerializedObject(this).getObject();
    }
	
	@Override
	public TechnicalInformation getTechnicalInformation() {
		// TODO Auto-generated method stub
		return null;
	}

}
