
package mulan.classifier.neural;


/**
 * Implements the linear activation function. The input is simply passed to the output. 
 * This activation function is commonly used for input units of networks, which serves 
 * as a place holders for input pattern and forwards them for processing. 
 * 
 * @author Jozef Vilcek
 */
public class ActivationLinear extends ActivationFunction {

	private static final long serialVersionUID = 4255801421493489832L;

	public double activate(final double input) {
		return input;
	}

	public double derivative(final double input) {
		throw new UnsupportedOperationException("Can't compute a derivative of the linear activation function."); 
	}

	public double getMax() {
		return Double.POSITIVE_INFINITY;
	}

	public double getMin() {
		return Double.NEGATIVE_INFINITY;
	}
}
