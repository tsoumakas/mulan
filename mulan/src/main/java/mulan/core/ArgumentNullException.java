/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
package mulan.core;

/**
 * This exception is raised when a null reference is passed to a method that does not accept
 * a null reference for an argument. 
 * 
 * @author Jozef Vilcek
 */
public class ArgumentNullException extends MulanRuntimeException {

    /** Version UID for serialization */
    private static final long serialVersionUID = -555866519789329786L;
    /** The name of the parameter which caused the exception */
    private final String paramName;

    /**
     * Creates a new instance of {@link ArgumentNullException} for specified parameter.
     *
     * @param paramName the name of the parameter which caused {@link ArgumentNullException}.
     */
    public ArgumentNullException(String paramName) {
        this(paramName, "Argument value can not be null.");
    }

    /**
     * Creates a new instance of {@link ArgumentNullException} with detailed message
     * for specified parameter.
     *
     * @param message the detailed message.
     * @param paramName the name of the parameter which caused {@link ArgumentNullException}.
     */
    public ArgumentNullException(String paramName, String message) {
        super(message);
        this.paramName = paramName;
    }

    @Override
    public String getMessage() {

        if (paramName != null && paramName.length() > 0) {
            StringBuilder message = new StringBuilder(super.getMessage());
            message.append(System.getProperty("line.separator"));
            message.append("Parameter name: ").append(paramName);
            return message.toString();
        } else {
            return super.getMessage();
        }
    }
}