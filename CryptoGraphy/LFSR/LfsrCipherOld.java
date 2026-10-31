import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;

/**
 * LFSR stream cipher: encrypts / decrypts a message by XOR with an LFSR
 * keystream.
 *
 * Polynomial table source:
 * R. Ward and T. Molteno, "Table of Linear Feedback Shift Registers"
 * (University of Otago, 2007) - entry with the fewest taps for each m.
 * Key = m, value = exponents of the primitive polynomial.
 * Example: 4 -> {4, 3, 0} means x^4 + x^3 + 1.
 */
public class LfsrCipherOld {

    // ----------------------------------------------------------------------
    // Constants
    // ----------------------------------------------------------------------
    private static final int MAX_TABLE_ROWS = 20; // rows of the clock table printed
    private static final int MAX_COLUMNS_M = 16; // above this m, flip-flops are printed compactly

    private static final Map<Integer, int[]> PRIMITIVE_POLYNOMIALS = new HashMap<>();

    private static void add(int m, int... exponents) {
        PRIMITIVE_POLYNOMIALS.put(m, exponents);
    }

    static {
        add(2, 2, 1, 0);
        add(3, 3, 2, 0);
        add(4, 4, 3, 0);
        add(5, 5, 3, 0);
        add(6, 6, 5, 0);
        add(7, 7, 6, 0);
        add(8, 8, 6, 5, 4, 0);
        add(9, 9, 5, 0);
        add(10, 10, 7, 0);
        add(11, 11, 9, 0);
        add(12, 12, 11, 8, 6, 0);
        add(13, 13, 12, 10, 9, 0);
        add(14, 14, 13, 11, 9, 0);
        add(15, 15, 14, 0);
        add(16, 16, 14, 13, 11, 0);
        add(17, 17, 14, 0);
        add(18, 18, 11, 0);
        add(19, 19, 18, 17, 14, 0);
        add(20, 20, 17, 0);
        add(21, 21, 19, 0);
        add(22, 22, 21, 0);
        add(23, 23, 18, 0);
        add(24, 24, 23, 21, 20, 0);
        add(25, 25, 22, 0);
        add(26, 26, 25, 24, 20, 0);
        add(27, 27, 26, 25, 22, 0);
        add(28, 28, 25, 0);
        add(29, 29, 27, 0);
        add(30, 30, 29, 26, 24, 0);
        add(31, 31, 28, 0);
        add(32, 32, 30, 26, 25, 0);
        add(33, 33, 20, 0);
        add(34, 34, 31, 30, 26, 0);
        add(35, 35, 33, 0);
        add(36, 36, 25, 0);
        add(37, 37, 36, 33, 31, 0);
        add(38, 38, 37, 33, 32, 0);
        add(39, 39, 35, 0);
        add(40, 40, 37, 36, 35, 0);
        add(41, 41, 38, 0);
        add(42, 42, 40, 37, 35, 0);
        add(43, 43, 42, 38, 37, 0);
        add(44, 44, 42, 39, 38, 0);
        add(45, 45, 44, 42, 41, 0);
        add(46, 46, 40, 39, 38, 0);
        add(47, 47, 42, 0);
        add(48, 48, 44, 41, 39, 0);
        add(49, 49, 40, 0);
        add(50, 50, 48, 47, 46, 0);
        add(51, 51, 50, 48, 45, 0);
        add(52, 52, 49, 0);
        add(53, 53, 52, 51, 47, 0);
        add(54, 54, 51, 48, 46, 0);
        add(55, 55, 31, 0);
        add(56, 56, 54, 52, 49, 0);
        add(57, 57, 50, 0);
        add(58, 58, 39, 0);
        add(59, 59, 57, 55, 52, 0);
        add(60, 60, 59, 0);
        add(61, 61, 60, 59, 56, 0);
        add(62, 62, 59, 57, 56, 0);
        add(63, 63, 62, 0);
        add(64, 64, 63, 61, 60, 0);
        add(65, 65, 47, 0);
        add(66, 66, 60, 58, 57, 0);
        add(67, 67, 66, 65, 62, 0);
        add(68, 68, 59, 0);
        add(69, 69, 67, 64, 63, 0);
        add(70, 70, 69, 67, 65, 0);
        add(71, 71, 65, 0);
        add(72, 72, 69, 63, 62, 0);
        add(73, 73, 48, 0);
        add(74, 74, 71, 70, 67, 0);
        add(75, 75, 74, 72, 69, 0);
        add(76, 76, 74, 72, 71, 0);
        add(77, 77, 75, 72, 71, 0);
        add(78, 78, 77, 76, 71, 0);
        add(79, 79, 70, 0);
        add(80, 80, 78, 76, 71, 0);
        add(81, 81, 77, 0);
        add(82, 82, 78, 76, 73, 0);
        add(83, 83, 81, 79, 76, 0);
        add(84, 84, 71, 0);
        add(85, 85, 84, 83, 77, 0);
        add(86, 86, 84, 81, 80, 0);
        add(87, 87, 74, 0);
        add(88, 88, 80, 79, 77, 0);
        add(89, 89, 51, 0);
        add(90, 90, 88, 87, 85, 0);
        add(91, 91, 90, 86, 83, 0);
        add(92, 92, 90, 87, 86, 0);
        add(93, 93, 91, 0);
        add(94, 94, 73, 0);
        add(95, 95, 84, 0);
        add(96, 96, 90, 87, 86, 0);
        add(97, 97, 91, 0);
        add(98, 98, 87, 0);
        add(99, 99, 95, 94, 92, 0);
        add(100, 100, 63, 0);
        add(101, 101, 100, 95, 94, 0);
        add(102, 102, 99, 97, 96, 0);
        add(103, 103, 94, 0);
        add(104, 104, 103, 94, 93, 0);
        add(105, 105, 89, 0);
        add(106, 106, 91, 0);
        add(107, 107, 105, 99, 98, 0);
        add(108, 108, 77, 0);
        add(109, 109, 107, 105, 104, 0);
        add(110, 110, 109, 106, 104, 0);
        add(111, 111, 101, 0);
        add(112, 112, 108, 106, 101, 0);
        add(113, 113, 104, 0);
        add(114, 114, 113, 112, 103, 0);
        add(115, 115, 110, 108, 107, 0);
        add(116, 116, 114, 111, 110, 0);
        add(117, 117, 116, 115, 112, 0);
        add(118, 118, 85, 0);
        add(119, 119, 111, 0);
        add(120, 120, 118, 114, 111, 0);
        add(121, 121, 103, 0);
        add(122, 122, 121, 120, 116, 0);
        add(123, 123, 121, 0);
        add(124, 124, 87, 0);
        add(125, 125, 120, 119, 118, 0);
        add(126, 126, 124, 122, 119, 0);
        add(127, 127, 126, 0);
        add(128, 128, 127, 126, 121, 0);
        add(129, 129, 124, 0);
        add(130, 130, 127, 0);
        add(131, 131, 129, 128, 123, 0);
        add(132, 132, 103, 0);
        add(133, 133, 131, 125, 124, 0);
        add(134, 134, 77, 0);
        add(135, 135, 124, 0);
        add(136, 136, 134, 133, 128, 0);
        add(137, 137, 116, 0);
        add(138, 138, 137, 131, 130, 0);
        add(139, 139, 136, 134, 131, 0);
        add(140, 140, 111, 0);
        add(141, 141, 140, 135, 128, 0);
        add(142, 142, 121, 0);
        add(143, 143, 141, 140, 138, 0);
        add(144, 144, 142, 140, 137, 0);
        add(145, 145, 93, 0);
        add(146, 146, 144, 143, 141, 0);
        add(147, 147, 145, 143, 136, 0);
        add(148, 148, 121, 0);
        add(149, 149, 142, 140, 139, 0);
        add(150, 150, 97, 0);
        add(151, 151, 148, 0);
        add(152, 152, 150, 149, 146, 0);
        add(153, 153, 152, 0);
        add(154, 154, 153, 149, 145, 0);
        add(155, 155, 151, 150, 148, 0);
        add(156, 156, 153, 151, 147, 0);
        add(157, 157, 155, 152, 151, 0);
        add(158, 158, 153, 152, 150, 0);
        add(159, 159, 128, 0);
        add(160, 160, 158, 157, 155, 0);
        add(161, 161, 143, 0);
        add(162, 162, 158, 155, 154, 0);
        add(163, 163, 160, 157, 156, 0);
        add(164, 164, 159, 158, 152, 0);
        add(165, 165, 162, 157, 156, 0);
        add(166, 166, 164, 163, 156, 0);
        add(167, 167, 161, 0);
        add(168, 168, 162, 159, 152, 0);
    }

    // ----------------------------------------------------------------------
    // LFSR
    // ----------------------------------------------------------------------

    /**
     * Fibonacci LFSR with flip-flops FF1 ... FFm (state[0] = FF1, state[m-1] =
     * FFm).
     *
     * Each clock:
     * output = FFm
     * feedback = XOR of the flip-flops whose index is a polynomial exponent
     * (the x^0 term is the feedback input, not a flip-flop)
     * shift : FF(i) <- FF(i-1), FF1 <- feedback
     */
    private static class Lfsr {
        private final int[] taps;
        private int[] state;

        Lfsr(int[] exponents, int[] initialVector) {
            this.taps = java.util.Arrays.stream(exponents).filter(e -> e > 0).toArray();
            this.state = initialVector.clone();
        }

        int[] getState() {
            return state.clone();
        }

        /** Advance one clock and return the output bit. */
        int clock() {
            int output = state[state.length - 1];
            int feedback = 0;
            for (int tap : taps) {
                feedback ^= state[tap - 1];
            }
            int[] next = new int[state.length];
            next[0] = feedback;
            System.arraycopy(state, 0, next, 1, state.length - 1);
            state = next;
            return output;
        }
    }

    // ----------------------------------------------------------------------
    // Helpers: polynomial, bits <-> text, formatting
    // ----------------------------------------------------------------------

    /** {4, 3, 0} -> "x^4 + x^3 + 1" */
    private static String polynomialToString(int[] exponents) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < exponents.length; i++) {
            if (i > 0)
                sb.append(" + ");
            int e = exponents[i];
            if (e == 0)
                sb.append("1");
            else if (e == 1)
                sb.append("x");
            else
                sb.append("x^").append(e);
        }
        return sb.toString();
    }

    /** String -> bits (UTF-8, 8 bits per byte, MSB first). */
    private static int[] textToBits(String text) {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        int[] bits = new int[bytes.length * 8];
        for (int i = 0; i < bytes.length; i++) {
            for (int b = 0; b < 8; b++) {
                bits[i * 8 + b] = (bytes[i] >> (7 - b)) & 1;
            }
        }
        return bits;
    }

    /** Bits -> bytes (8 bits per byte, MSB first). */
    private static byte[] bitsToBytes(int[] bits) {
        byte[] bytes = new byte[bits.length / 8];
        for (int i = 0; i < bytes.length; i++) {
            int value = 0;
            for (int b = 0; b < 8; b++) {
                value = (value << 1) | bits[i * 8 + b];
            }
            bytes[i] = (byte) value;
        }
        return bytes;
    }

    /** Ciphertext bits -> readable string (non-printable bytes shown as '.'). */
    private static String bitsToDisplayString(int[] bits) {
        StringBuilder sb = new StringBuilder();
        for (byte b : bitsToBytes(bits)) {
            int c = b & 0xFF;
            sb.append(c >= 32 && c < 127 ? (char) c : '.');
        }
        return sb.toString();
    }

    private static String bytesToHex(byte[] bytes) {
        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) {
            sb.append(String.format("%02x", b & 0xFF));
        }
        return sb.toString();
    }

    /** {1, 0, 1} -> "101" */
    private static String bitsToStr(int[] bits) {
        StringBuilder sb = new StringBuilder();
        for (int bit : bits)
            sb.append(bit);
        return sb.toString();
    }

    /** Flip-flop values as text; spaced for small m, compact for large m. */
    private static String formatState(int[] state) {
        if (state.length > MAX_COLUMNS_M) {
            return bitsToStr(state);
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < state.length; i++) {
            if (i > 0)
                sb.append(' ');
            sb.append(state[i]);
        }
        return sb.toString();
    }

    /** Column width needed to print m flip-flops. */
    private static int stateWidth(int m) {
        String header = "Flip-flops (FF1..FF" + m + ")";
        int values = (m <= MAX_COLUMNS_M) ? 2 * m - 1 : m;
        return Math.max(header.length(), values);
    }

    private static void printTableHeader(int m, String col3, String col4, String col5) {
        int width = stateWidth(m);
        String header = String.format("%5s | %-" + width + "s | %-3s | %-6s | %s",
                "Clock", "Flip-flops (FF1..FF" + m + ")", col3, col4, col5);
        System.out.println(header);
        System.out.println("-".repeat(header.length()));
    }

    private static void printTableRow(int clock, int[] stateBefore, int m, int out, int mid, int last) {
        System.out.printf("%5d | %-" + stateWidth(m) + "s | %-3d | %-6d | %d%n",
                clock, formatState(stateBefore), out, mid, last);
    }

    // ----------------------------------------------------------------------
    // User input / initial vector
    // ----------------------------------------------------------------------

    private static String askMessage(Scanner in) {
        while (true) {
            System.out.print("Enter the message: ");
            String message = in.nextLine();
            if (!message.isEmpty())
                return message;
            System.out.println("  Message cannot be empty, try again.");
        }
    }

    /** Read m until it exists in the polynomial table. */
    private static int askM(Scanner in) {
        while (true) {
            System.out.print("Enter m (number of flip-flops, 2.." + 168 + "): ");
            String raw = in.nextLine().trim();
            try {
                int m = Integer.parseInt(raw);
                if (PRIMITIVE_POLYNOMIALS.containsKey(m))
                    return m;
            } catch (NumberFormatException ignored) {
                // falls through to the error message below
            }
            System.out.println("  m = '" + raw + "' is not in the polynomial table, try again.");
        }
    }

    /** Random vector of m bits (0/1 only), guaranteed not to be all zeros. */
    private static int[] generateInitialVector(int m) {
        Random random = new Random();
        while (true) {
            int[] vector = new int[m];
            boolean hasOne = false;
            for (int i = 0; i < m; i++) {
                vector[i] = random.nextInt(2);
                if (vector[i] == 1)
                    hasOne = true;
            }
            if (hasOne)
                return vector;
        }
    }

    // ----------------------------------------------------------------------
    // Encryption
    // ----------------------------------------------------------------------

    /** Encrypt the message with the LFSR keystream. Returns the cipher bits. */
    private static int[] encrypt(String message, int m, int[] initialVector) {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("ENCRYPTION");
        System.out.println("=".repeat(60));

        int[] exponents = PRIMITIVE_POLYNOMIALS.get(m);
        Lfsr lfsr = new Lfsr(exponents, initialVector);
        int[] messageBits = textToBits(message);
        int[] cipherBits = new int[messageBits.length];

        System.out.println("Polynomial     : " + polynomialToString(exponents));
        System.out.println("Message bits   : " + messageBits.length + " bits");
        System.out.println("\nClock table (first " + MAX_TABLE_ROWS + " of " + messageBits.length + " clocks):");
        printTableHeader(m, "Out", "Msg", "Cipher");

        for (int i = 0; i < messageBits.length; i++) {
            int clock = i + 1;
            int[] stateBefore = lfsr.getState();
            int keyBit = lfsr.clock();
            cipherBits[i] = messageBits[i] ^ keyBit;

            if (clock <= MAX_TABLE_ROWS) {
                printTableRow(clock, stateBefore, m, keyBit, messageBits[i], cipherBits[i]);
            }
        }

        System.out.println("\nCiphertext (binary):");
        System.out.println(bitsToStr(cipherBits));
        System.out.println("Ciphertext (string): " + bitsToDisplayString(cipherBits));
        System.out.println("Ciphertext (hex)   : " + bytesToHex(bitsToBytes(cipherBits)));
        return cipherBits;
    }

    // ----------------------------------------------------------------------
    // Decryption
    // ----------------------------------------------------------------------

    /**
     * Decrypt the cipher bits. The keystream from encryption is NOT reused:
     * the LFSR is rebuilt from the same polynomial + initial vector and the
     * keystream is generated again from scratch.
     */
    private static String decrypt(int[] cipherBits, int m, int[] initialVector) {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("DECRYPTION");
        System.out.println("=".repeat(60));

        int[] exponents = PRIMITIVE_POLYNOMIALS.get(m);
        Lfsr lfsr = new Lfsr(exponents, initialVector); // fresh LFSR, reset to the initial vector
        int[] plainBits = new int[cipherBits.length];

        System.out.println("Polynomial     : " + polynomialToString(exponents));
        System.out.println("LFSR reset to  : " + formatState(initialVector));
        System.out.println("\nClock table (first " + MAX_TABLE_ROWS + " of " + cipherBits.length + " clocks):");
        printTableHeader(m, "Out", "Cipher", "Plain");

        for (int i = 0; i < cipherBits.length; i++) {
            int clock = i + 1;
            int[] stateBefore = lfsr.getState();
            int keyBit = lfsr.clock();
            plainBits[i] = cipherBits[i] ^ keyBit;

            if (clock <= MAX_TABLE_ROWS) {
                printTableRow(clock, stateBefore, m, keyBit, cipherBits[i], plainBits[i]);
            }
        }

        String plainText = new String(bitsToBytes(plainBits), StandardCharsets.UTF_8);
        System.out.println("\nDecrypted bits:");
        System.out.println(bitsToStr(plainBits));
        System.out.println("Decrypted text : " + plainText);
        return plainText;
    }

    // ----------------------------------------------------------------------
    // Main
    // ----------------------------------------------------------------------

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in, StandardCharsets.UTF_8);

        System.out.println("LFSR Encryption & Decryption");
        System.out.println("-".repeat(28));
        String message = askMessage(in);
        int m = askM(in);

        int[] initialVector = generateInitialVector(m);
        System.out.println("\nm              : " + m);
        System.out.println("Initial vector : " + bitsToStr(initialVector));

        int[] cipherBits = encrypt(message, m, initialVector);
        String decrypted = decrypt(cipherBits, m, initialVector);

        System.out.println("\n" + "=".repeat(60));
        System.out.println("Original message : " + message);
        System.out.println("Decrypted message: " + decrypted);
        System.out.println(decrypted.equals(message) ? "MATCH" : "MISMATCH");
    }
}
