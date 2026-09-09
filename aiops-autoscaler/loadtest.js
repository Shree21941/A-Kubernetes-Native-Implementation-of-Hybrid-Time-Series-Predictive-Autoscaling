import http from 'k6/http';
import { sleep } from 'k6';

// Ye script "rush hour" jaisa pattern banata hai:
// - Pehle kam users (baseline)
// - Phir achanak spike (rush hour burst)
// - Phir wapas normal
// - Ye cycle repeat hota hai taaki forecasting model ko "pattern" seekhne layak data mile

export const options = {
  stages: [
    { duration: '2m', target: 5 },    // Baseline: 5 users, 2 min
    { duration: '1m', target: 50 },   // Sudden spike: 50 users, 1 min (rush hour!)
    { duration: '2m', target: 50 },   // Rush hour continues, 2 min
    { duration: '1m', target: 5 },    // Rush khatam, wapas normal
    { duration: '2m', target: 5 },    // Baseline phir se, 2 min
    { duration: '1m', target: 80 },   // Bada spike: 80 users, 1 min
    { duration: '2m', target: 80 },   // Bada rush continues
    { duration: '1m', target: 5 },    // Normal pe wapas
  ],
};

const BASE_URL = 'http://localhost:8080';

export default function () {
  // Homepage visit karo
  http.get(`${BASE_URL}/`);
  sleep(1);

  // Product page visit karo (random product)
  const products = [
    '0PUK6V6EV0',
    '1YMWWN1N4O',
    '2ZYFJ3GM2N',
    '66VCHSJNUP',
    '6E92ZMYYFZ',
  ];
  const productId = products[Math.floor(Math.random() * products.length)];
  http.get(`${BASE_URL}/product/${productId}`);
  sleep(1);

  // Cart mein add karo (checkout flow simulate karo)
  http.get(`${BASE_URL}/cart`);
  sleep(1);
}
