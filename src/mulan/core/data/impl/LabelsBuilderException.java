package mulan.core.data.impl;

/**
 * Exception is raised by {@link LabelsBuilder} to indicate an error when creating 
 * {@link LabelsMetaDataImpl} instance form specified source.
 * 
 * @author Jozef Vilcek
 * @see LabelsBuilder
 */
public class LabelsBuilderException extends Exception {

	private static final long serialVersionUID = 2161709838882541792L;


	public LabelsBuilderException() {
    	super();
    }

    public LabelsBuilderException(String message) {
    	super(message);
    }

    public LabelsBuilderException(String message, Throwable cause) {
        super(message, cause);
    }

    public LabelsBuilderException(Throwable cause) {
        super(cause);
    }
}
