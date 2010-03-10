package meta;

import mulan.classifier.lazy.BRkNN;
import mulan.classifier.meta.RAkELd;

public class RAkELdTest extends MultiLabelMetaLearnerTest {

	@Override
	public void setUp() throws Exception {
		learner = new RAkELd(new BRkNN(10));
	}

}
