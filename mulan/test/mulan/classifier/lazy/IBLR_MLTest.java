package mulan.classifier.lazy;

import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelLearnerTestBase;

import org.junit.Before;

public class IBLR_MLTest extends MultiLabelLearnerTestBase {
	
	private static final int DEFAULT_numOfNeighbors = 10; 
	
	private IBLR_ML learner;
	
	@Override
	protected MultiLabelLearnerBase getLearner() {
		return learner;
	}
	
	@Before
	public void setUp(){
		learner = new  IBLR_ML(DEFAULT_numOfNeighbors);
	}

}
