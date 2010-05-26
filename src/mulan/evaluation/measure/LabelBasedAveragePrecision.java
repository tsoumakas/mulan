package mulan.evaluation.measure;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * 
 * 
 * @author Eleftherios Spyromitros Xioufis
 * 
 */
public abstract class LabelBasedAveragePrecision extends ConfidenceMeasureBase {

	protected int numOfLabels;

	List<ConfidenceActual>[] confact;

	public LabelBasedAveragePrecision(int numOfLabels) {
		this.numOfLabels = numOfLabels;
		confact = new ArrayList[numOfLabels];
		for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
			confact[labelIndex] = new ArrayList<ConfidenceActual>();
		}
	}

	public double updateInternal2(double[] confidences, boolean[] truth) {
		for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
			boolean actual = truth[labelIndex];
			// boolean predicted = bipartition[labelIndex];
			double confidence = confidences[labelIndex];
			// another metric...
			// if (predicted) {
			confact[labelIndex].add(new ConfidenceActual(confidence, actual));
			// }
		}

		return 0;
	}

	public void reset() {
		for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
			confact[labelIndex].clear();
		}
	}

	public class ConfidenceActual implements Comparable,Serializable {

		boolean actual;
		double confidence;

		public ConfidenceActual(double confidence, boolean actual) {
			this.actual = actual;
			this.confidence = confidence;
		}

		public boolean getActual() {
			return actual;
		}

		public double getConfidence() {
			return confidence;
		}

		@Override
		public int compareTo(Object o) {
			if (this.confidence > ((ConfidenceActual) o).confidence) {
				return 1;
			} else if (this.confidence < ((ConfidenceActual) o).confidence) {
				return -1;
			} else {
				return 0;
			}
		}
	}
}
