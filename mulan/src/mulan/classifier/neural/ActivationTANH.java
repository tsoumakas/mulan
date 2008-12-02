
package mulan.classifier.neural;

/**
 * Implements the hyperbolic tangent activation function.
 * The function output values are from interval <-1,1>.
 * 
 * @author Jozef Vilcek
 */
public class ActivationTANH extends ActivationFunction {

	private static final long serialVersionUID = -8707244320811304601L;
	public final static double MAX = +1.0;
	public final static double MIN = -1.0;
	
	public double activate(final double input) {
        return 2.0 / (1.0 + Math.exp(-2.0 * input)) - 1.0; 
	}

	public double derivative(final double input) {
		return 1.0 - Math.pow(activate(input), 2.0);
	}

	public double getMax() {
		return MAX;
	}

	public double getMin() {
		return MIN;
	}
	
}
