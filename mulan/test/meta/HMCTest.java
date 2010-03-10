package meta;

import mulan.classifier.lazy.BRkNN;
import mulan.classifier.meta.HMC;

public class HMCTest extends MultiLabelMetaLearnerTest {

	@Override
	public void setUp() throws Exception {
		learner = new  HMC(new BRkNN(10));
	}

}
