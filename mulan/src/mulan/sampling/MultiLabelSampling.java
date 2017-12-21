package mulan.sampling;

import java.util.Random;

import mulan.data.MultiLabelInstances;

public abstract class MultiLabelSampling {
	
	protected double P=0.3; //Percentage of instances to be deleted
	
	protected Random rnd;
	
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
		rnd=new Random(seed);
	}

	
	
	public abstract MultiLabelInstances build(MultiLabelInstances mlDataset) throws Exception;
}
