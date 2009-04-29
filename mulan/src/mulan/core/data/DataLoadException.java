package mulan.core.data;

import mulan.core.MulanRuntimeException;

/**
 * The exception is thrown to indicate an error while loading the data.
 * 
 * @author Jozef Vilcek
 */
public class DataLoadException extends MulanRuntimeException {

	private static final long serialVersionUID = 1102055196162204723L;

	/**
	 * Creates a new instance of {@link DataLoadException} with detail mesage.
	 * @param message the detail message
	 */
	public DataLoadException(String message){
		super(message);
	}
	
	/**
	 * Creates a new instance of {@link DataLoadException} with detail message 
	 * and nested exception.
	 * 
	 * @param message the detail message
	 * @param cause the nested exception
	 */
	public DataLoadException(String message, Throwable cause){
		super(message, cause);
	}
}
