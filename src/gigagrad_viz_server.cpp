#include "httplib.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;


void StartServer() {
    httplib::Server svr;

    svr.Get("/test-tensor", [](const httplib::Request &, httplib::Response &res) {
        const json test_tensor = {
            {"shape", {1, 6, 6, 3}},
            {"buffer", {
                1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6,
                2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7,
                3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8,
                3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8,
                3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8,
                3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8
            }}
        };
        res.set_content(test_tensor.dump(), "application/json");
    });

    svr.listen("0.0.0.0", 8080);
}

int main()
{
    StartServer();
    return 0;
}
