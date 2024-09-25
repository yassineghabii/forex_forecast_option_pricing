import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AboutUsBoxComponent } from './about-us-box.component';

describe('AboutUsBoxComponent', () => {
  let component: AboutUsBoxComponent;
  let fixture: ComponentFixture<AboutUsBoxComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ AboutUsBoxComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(AboutUsBoxComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
