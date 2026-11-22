package task;

import java.util.ArrayList;
import java.util.Arrays;

/** New independent checks of public semantics, not the missing original oracle. */
public final class ConnectionCases {
    private static int failures = 0;

    private static void runCase(String name, String[] present,
                                boolean wifiSuccess, boolean mobileSuccess,
                                String expectedType) {
        DeviceAPI.API = new DeviceAPI() {
            public ArrayList<String> getAvailableConnections() {
                return new ArrayList<String>(Arrays.asList(present));
            }
        };
        ConnectionAPI.API = new ConnectionAPI() {
            public boolean isWifiAvailable() { return wifiSuccess; }
            public boolean isMobileDataAvailable() { return mobileSuccess; }
        };
        Device device = new Device();
        boolean threw = false;
        boolean exactFailure = false;
        String actual = "none";
        String exception = null;
        try {
            device.connect();
            if (device.connection != null) {
                actual = device.connection.getClass().getSimpleName();
            }
        } catch (java.lang.RuntimeException error) {
            threw = true;
            exactFailure = error.getClass() == java.lang.RuntimeException.class;
            exception = error.getClass().getSimpleName();
        }
        // This independent oracle requires the explicitly signalled failure class.
        // Accidental subclasses such as NullPointerException are never successes.
        boolean pass = expectedType.equals("RuntimeException")
            ? threw && exactFailure
            : !threw && expectedType.equals(actual);
        if (!pass) failures++;
        // All emitted values are fixed fixture identifiers or Java class names.
        System.out.println("{\"case\":\"" + name + "\",\"passed\":" + pass
            + ",\"expected\":\"" + expectedType + "\",\"actual\":\""
            + (threw ? exception : actual) + "\"}");
    }

    public static void main(String[] args) {
        runCase("wifi_priority_despite_list_order", new String[]{"mobiledata", "wifi"}, true, true, "Wifi");
        runCase("wifi_only_success", new String[]{"wifi"}, true, false, "Wifi");
        runCase("mobile_only_success", new String[]{"mobiledata"}, true, true, "MobileData");
        runCase("failed_wifi_falls_back_to_mobile", new String[]{"wifi", "mobiledata"}, false, true, "MobileData");
        runCase("no_available_types_throws", new String[]{}, true, true, "RuntimeException");
        runCase("all_available_attempts_fail", new String[]{"wifi", "mobiledata"}, false, false, "RuntimeException");
        runCase("unavailable_mobile_is_not_fallback", new String[]{"wifi"}, false, true, "RuntimeException");
        runCase("unavailable_wifi_is_not_fallback", new String[]{"mobiledata"}, true, false, "RuntimeException");
        if (failures != 0) System.exit(1);
    }
}
