import http from 'k6/http';
import { sleep } from 'k6';

// Realistic traffic mix for Online Boutique -- this app is server-rendered,
// not a REST API, so the HTTP verbs it genuinely exercises are GET (browsing)
// and POST (cart mutations, checkout, currency change). Faking PUT/DELETE
// against endpoints that don't exist would just 404 instantly and add no
// real backend load -- useless for the model. This mix covers every
// verb+endpoint combination the app actually has.

export const options = {
  stages: [
    { duration: '2m', target: 5 },     // baseline
    { duration: '1m', target: 50 },    // spike
    { duration: '2m', target: 50 },    // spike hold
    { duration: '1m', target: 5 },     // cooldown
    { duration: '2m', target: 5 },     // baseline
    { duration: '1m', target: 80 },    // big spike
    { duration: '2m', target: 80 },    // big spike hold
    { duration: '1m', target: 5 },     // cooldown
    { duration: '2m', target: 5 },     // baseline (pre-DDoS calm, for contrast)
    { duration: '30s', target: 400 },  // DDoS-style overwhelming ramp
    { duration: '1m30s', target: 400 },// sustained overload
    { duration: '1m', target: 5 },     // sudden drop -- watch scale-down behaviour
  ],
};

const BASE_URL = 'http://localhost:8080';

const products = [
  '0PUK6V6EV0', '1YMWWN1N4O', '2ZYFJ3GM2N',
  '66VCHSJNUP', '6E92ZMYYFZ', '9SIQT8TOJO', 'L9ECAV7KIM', 'LS4PSXUNUM',
];
const currencies = ['USD', 'EUR', 'JPY', 'GBP', 'CAD'];

function randOf(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

export default function () {
  const roll = Math.random();

  if (roll < 0.45) {
    // --- Browse: GET home + a product page ---
    http.get(`${BASE_URL}/`);
    sleep(0.5);
    http.get(`${BASE_URL}/product/${randOf(products)}`);

  } else if (roll < 0.65) {
    // --- Add to cart: POST form data ---
    http.post(`${BASE_URL}/cart`, {
      product_id: randOf(products),
      quantity: String(1 + Math.floor(Math.random() * 3)),
    });
    sleep(0.3);
    http.get(`${BASE_URL}/cart`);

  } else if (roll < 0.8) {
    // --- Change currency: POST ---
    http.post(`${BASE_URL}/setCurrency`, {
      currency_code: randOf(currencies),
    });
    sleep(0.3);
    http.get(`${BASE_URL}/`);

  } else if (roll < 0.95) {
    // --- Full checkout flow: add to cart then POST checkout ---
    http.post(`${BASE_URL}/cart`, {
      product_id: randOf(products),
      quantity: '1',
    });
    sleep(0.3);
    http.post(`${BASE_URL}/cart/checkout`, {
      email: 'loadtest@example.com',
      street_address: '1600 Amphitheatre Pkwy',
      zip_code: '94043',
      city: 'Mountain View',
      state: 'CA',
      country: 'United States',
      credit_card_number: '4432801561520454',
      credit_card_expiration_month: '1',
      credit_card_expiration_year: '2028',
      credit_card_cvv: '672',
    });

  } else {
    // --- Empty cart: POST ---
    http.post(`${BASE_URL}/cart/empty`);
    sleep(0.2);
    http.get(`${BASE_URL}/`);
  }

  sleep(0.5 + Math.random());
}
