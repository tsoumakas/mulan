package mulan.sampling;

import mulan.data.MultiLabelInstances;

/***
 * Do not apply any sampling strategy, just return the input data set
 * 
 * @author Bin Liu
 * @version 2019.3.19
 *
 */
public class NoProcess extends MultiLabelSampling{

	@Override
	public MultiLabelInstances build(MultiLabelInstances mlDataset) throws Exception {
		// TODO Auto-generated method stub
		return mlDataset;
	}

}
